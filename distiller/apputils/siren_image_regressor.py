from pprint import pprint

import copy
import math
import time
import os
import sys
import logging
import json
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
# import parser
from functools import partial
import argparse
import distiller
import distiller.apputils
from distiller.apputils.performance_tracker import SparsityMSETracker
from distiller.data_loggers import *
import distiller.quantization
import distiller.models
from distiller.models import create_model
from distiller.utils import float_range_argparse_checker as float_range


import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

from distiller.pruning.automated_gradual_pruner import AutomatedGradualPruner

import distiller.apputils.siren_utils.siren_init_utils
import distiller.apputils.siren_utils.siren_train_val_test_utils

# Logger handle
msglogger = logging.getLogger()
PRUNE_DETAILS = {}
TOLL = 2.0

# ----------------------------------------------------------------------------------------------- #
# SirenRegressorCompressor: Class Definition
# ----------------------------------------------------------------------------------------------- #
class SirenRegressorCompressor(object):
    """Base class for applications that want to compress image regressor.

    This class performs boiler-plate code used in image-regressor compression:
        - Command-line arguments handling
        - Logger configuration
        - Data loading
        - Checkpoint handling
        - Regressor training, verification and testing
    """
    def __init__(self, args, script_dir):
        # Configure instance of such a class to work properly downstream while training is performed,
        # as well as validation or test when also these are requested.
        try: self.args = copy.deepcopy(args)
        except: self.args = args

        self.test_mode_on = False
        self.args = self._infer_implicit_args(self.args)

        self.logdir = distiller.apputils.siren_utils.siren_init_utils._init_logger(self.args, script_dir, msglogger)
        distiller.apputils.siren_utils.siren_init_utils._config_determinism(self.args, msglogger)
        distiller.apputils.siren_utils.siren_init_utils._config_compute_device(self.args, msglogger)
        
        # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
        # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
        if not self.logdir:
            self.pylogger = self.tflogger = NullLogger()
        else:
            self.tflogger = TensorBoardLogger(msglogger.logdir)
            self.pylogger = PythonLogger(msglogger)
        (self.model, self.compression_scheduler, self.optimizer, 
             self.start_epoch, self.ending_epoch) =  distiller.apputils.siren_utils.siren_init_utils._init_learner(self.args, msglogger)

        # Define loss function (criterion)
        self.criterion = nn.MSELoss().to(self.args.device)
        self.train_loader, self.val_loader, self.test_loader = (None, None, None)
        self.activations_collectors = create_activation_stats_collectors(
            self.model, *self.args.activation_stats)
        
        # Create an object to record scores and mentrics while training a model base on siren net.
        try:
            self.performance_tracker = distiller.apputils.SparsityMSETracker(self.args.num_best_scores)
        except Exception as _:
            self.performance_tracker = SparsityMSETracker(self.args.num_best_scores)
        
        # Check whether to setup an object to keep track
        # when it's time to stop training since target sparsity level
        # has been reached.
        if self.args.target_sparsity is not None:
            self.early_stopping_agp = EarlyStoppingAGP(
                target_sparsity=self.args.target_sparsity, toll=self.args.toll_sparsity,
                patience=self.args.patience_sparsity, trail_epochs=self.args.trail_epochs)
        else:
            self.early_stopping_agp = None
        
        # Check whether to setup an object to keep track
        # when it's necessary to save a middle prune level
        # reached while pruning a model.
        if self.args.mid_target_sparsities != []:
            self.save_mid_pr = SaveMiddlePruneRate(middle_prune_rates=self.args.mid_target_sparsities)
            msglogger.info(f"Created SaveMiddlePruneRate from: {str(self.args.mid_target_sparsities)}")
        else:
            self.save_mid_pr = None
        
    
    def load_datasets(self):
        """Load the datasets"""
        global msglogger
        msglogger.info("Loading datasets...")
        if not all((self.train_loader, self.val_loader, self.test_loader)):
            self.train_loader, self.val_loader, self.test_loader = load_data(self.args)
        msglogger.info("Datasets correctly loaded.")
        return self.data_loaders


    def get_dataframe_model(self,):
        _, total, df = distiller.weights_sparsity_tbl_summary(self.model, return_total_sparsity=True, return_df=True)
        return df



    @property
    def data_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader


    @staticmethod
    def _infer_implicit_args(args):
        # Infer the dataset from the model name
        if not hasattr(args, 'verbose'):
            args.verbose = False
        return args


    @staticmethod
    def mock_args():
        """Generate a Namespace based on default arguments"""
        return SirenRegressorCompressor._infer_implicit_args(
            distiller.apputils.siren_utils.siren_init_utils.init_regressor_compression_arg_parser().parse_args(['fictive_required_arg',]))


    @classmethod
    def mock_classifier(cls):
        return cls(cls.mock_args(), '')


    def get_input_target(self, kind = None):
        if kind and kind == 'train':
            data_loader = self.train_loader.sampler
        elif kind and kind == 'val':
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader
        
        total_samples = len(data_loader.sampler)
        batch_size = data_loader.batch_size

        inputs, target = next(iter(data_loader))
        inputs, target = inputs.to(self.args.device), target.to(self.args.device)
        return inputs, target, total_samples, batch_size


    def train_one_epoch(self, epoch, verbose=True, is_last_epoch = False):
        """Train for one epoch"""
        # self.load_datasets()

        inputs, target, total_samples, batch_size = self.get_input_target(kind='train')

        with collectors_context(self.activations_collectors["train"]) as collectors:
            loss = \
                distiller.apputils.siren_utils.siren_train_val_test_utils.train(
                    # self.train_loader,
                    inputs, target, total_samples, batch_size, 
                    self.model,
                    self.criterion, self.optimizer, 
                    epoch, self.compression_scheduler,
                    loggers=[self.tflogger, self.pylogger], args=self.args, is_last_epoch = is_last_epoch,
                    early_stopping_agp=self.early_stopping_agp,
                    save_mid_pr=self.save_mid_pr)
            
            distiller.log_activation_statistics(epoch, "train", loggers=[self.tflogger],
                                                collector=collectors["sparsity"])
            # if self.args.compress and epoch >= 0 and epoch % self.args.print_freq == 0:
            if epoch >= 0 and epoch % self.args.print_freq == 0:
                distiller.log_weights_sparsity(self.model, epoch, [self.tflogger, self.pylogger])
            if self.args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(self.model, 
                                                                    self.compression_scheduler))
        return loss


    def train_validate_with_scheduling(self, epoch, validate=True, verbose=True, is_last_epoch = False):
        if self.compression_scheduler:
            self.compression_scheduler.on_epoch_begin(epoch)

        loss = self.train_one_epoch(epoch, verbose, is_last_epoch = is_last_epoch)
        if validate:
            loss, psnr_score, ssim_score = self.validate_one_epoch(epoch, verbose, is_last_epoch = is_last_epoch)

        if self.compression_scheduler:
            self.compression_scheduler.on_epoch_end(epoch, self.optimizer, 
                                                    metrics={'min': loss,})
        return loss, psnr_score, ssim_score


    def validate_one_epoch(self, epoch, verbose=True, is_last_epoch = False):
        """Evaluate on validation set"""
        # self.load_datasets()
        inputs_val, target_Val, total_samples_val, batch_size_val = self.get_input_target(kind='val')
        with collectors_context(self.activations_collectors["valid"]) as collectors:
            vloss, vpsnr, vssim = distiller.apputils.siren_utils.siren_train_val_test_utils.validate(
                # self.val_loader, self.model, self.criterion, 
                inputs_val, target_Val, total_samples_val, batch_size_val, \
                self.model, self.criterion, 
                [self.pylogger], self.args, epoch, is_last_epoch = is_last_epoch)
            distiller.log_activation_statistics(epoch, "valid", loggers=[self.tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        if verbose:
            stats = ('Performance/Validation/',
            OrderedDict([('Loss', vloss),
                ('PSNR', vpsnr),
                ('SSIM', vssim),
            ]))
            distiller.log_training_progress(stats, None, epoch, steps_completed=0,
                                            total_steps=1, log_freq=1, loggers=[self.tflogger])
        return vloss, vpsnr, vssim


    def _finalize_epoch(self, epoch, mse, psnr_score, ssim_score, is_last_epoch=False, prune_details={}):
        """Finalize data obtained for current epoch if it is necessary."""
        # def _finalize_epoch(self, epoch, mse, psnr_score, ssim_score, prune_details = {}):
        # Update the list of top scores achieved so far, and save the checkpoint

        is_one_to_save_pruned = False
        # if self.save_mid_pr is not None: is_one_to_save_pruned = self.save_mid_pr.is_one_to_save()

        self.performance_tracker.step(
            self.model,
            epoch,
            mse=mse,
            psnr_score=psnr_score, ssim_score=ssim_score)
        # if epoch >= 0 and epoch % self.args.print_freq == 0: _log_best_scores(self.performance_tracker, msglogger)
        best_score = self.performance_tracker.best_scores()[0]
        is_best = epoch == best_score.epoch
        checkpoint_extras = {'current_mse': mse,
                             'current_psnr_score': psnr_score,
                             'best_mse': best_score.mse,
                             'best_psnr_score': best_score.psnr_score,
                             'best_ssim_score': best_score.ssim_score,
                             'best_epoch': best_score.epoch}
        if msglogger.logdir:
            is_mid_ckpt = False
            if self.args.save_mid_ckpts != []:
                is_mid_ckpt = epoch in self.args.save_mid_ckpts
            
            # prune_details = distiller.apputils.siren_utils.siren_train_val_test_utils.get_prune_detail()
            distiller.apputils.save_checkpoint(
                epoch=epoch,
                arch=self.args.arch, model=self.model, optimizer=self.optimizer, scheduler=self.compression_scheduler, \
                extras=checkpoint_extras, \
                name=self.args.name, dir=msglogger.logdir, # freq_ckpt=self.args.print_freq,\
                freq_ckpt=10000,
                is_best=is_best, \
                is_mid_ckpt = is_mid_ckpt, \
                # is_last_epoch = is_last_epoch,
                is_one_to_save_pruned=is_one_to_save_pruned, \
                save_mid_pr_obj=self.save_mid_pr,
                prune_details=prune_details \
            )


    def run_training_loop_with_scheduler(self,):
        """Train/Val process carried out by means of a scheduler instance that manages all operations
        needed for correctly taking care of model's compression workflow.
        """
        global msglogger
        """
        def _log_training_progress(loggers=[self.tflogger]):
            # Log some statistics

            # _, _, df = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True, return_df=True)
            stats_dict = OrderedDict()
            for loss_name, meter in losses.items(): stats_dict[loss_name] = meter.mean
            stats_dict['LR'] = self.optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time # batch_time.mean
            stats = ('Performance/Training/', stats_dict)

            params = self.model.named_parameters() if self.args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            # epoch, steps_completed,
                                            epoch, 1,
                                            math.ceil(total_samples / batch_size), self.args.print_freq,
                                            loggers=loggers)
        """

        """
        def _log_validation_progress(loggers=[self.tflogger]):
            # stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),])
            # stats_dict = OrderedDict([('Loss', loss)])
            #if not _is_earlyexit(args): # stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),])
            # else:
            # stats_dict = OrderedDict()
            for exitnum in range(args.num_exits):
                la_string = 'LossAvg' + str(exitnum)
                stats_dict[la_string] = args.losses_exits[exitnum].mean
        
            stats = ('Performance/Validation/', stats_dict)
            distiller.log_training_progress(stats, None, epoch, 1, # steps_completed,
                                            total_samples_val / batch_size_val, self.args.print_freq, [self.pylogger])
            
            stats = ('Performance/Validation/',
                OrderedDict([('Loss', loss), # vloss
                    ('PSNR', psnr_score), # vpsnr
                    ('SSIM', ssim_score), # vssim
                ]))
            distiller.log_training_progress(stats, None, epoch, steps_completed=0,
                                            total_steps=1, log_freq=1, loggers=loggers)
        """
        # ---------------------- Prepare data ---------------------- #
        prune_details = {}
        total_samples = len(self.train_loader.sampler)
        batch_size = self.train_loader.batch_size

        inputs, target = next(iter(self.train_loader))
        inputs, target = inputs.to(self.args.device), target.to(self.args.device)

        # total_samples_val = len(self.val_loader.sampler)
        # batch_size_val = self.val_loader.batch_size

        inputs_val, target_val = next(iter(self.val_loader))
        inputs_val, target_val = inputs_val.to(self.args.device), target_val.to(self.args.device)
        
        loggers = [self.tflogger, self.pylogger]

        # ---------------------- Loop: train_validate_with_scheduling ---------------------- #
        for epoch in range(self.start_epoch, self.ending_epoch):
            # is_last_epoch = epoch == self.ending_epoch - 1

            # [On Epoch Begin] => Distiller framework operats by means of scheduler instance
            # as main orchestrator of the model's compression process.
            self.compression_scheduler.on_epoch_begin(epoch)
            
            # ---------------------- train_one_epoch ---------------------- #
            # loss, psnr_score, ssim_score = self.train_validate_with_scheduling(epoch, is_last_epoch = is_last_epoch)
            with collectors_context(self.activations_collectors["train"]) as collectors:
                losses, batch_time = \
                    distiller.apputils.siren_utils.siren_train_val_test_utils.train_via_scheduler(
                        # self.train_loader,
                        # loggers=[self.tflogger, self.pylogger], is_last_epoch = is_last_epoch, early_stopping_agp=self.early_stopping_agp, save_mid_pr=self.save_mid_pr, msglogger=msglogger, args=self.args,
                        inputs, target, total_samples, batch_size, \
                        self.model, \
                        self.criterion, self.optimizer, \
                        epoch, self.compression_scheduler)
                
                # Train: log activation/weigth-sparsity stats 
                # ------------------------------------------#
                """
                distiller.log_activation_statistics(
                    epoch, \
                    "train", \
                    loggers=[self.tflogger], \
                    collector=collectors["sparsity"])
                
                distiller.log_weights_sparsity(
                    self.model, \
                    epoch, \
                    [self.tflogger])
                """
                # if self.args.compress and epoch >= 0 and epoch % self.args.print_freq == 0:
                # if epoch >= 0 and epoch % self.args.print_freq == 0: # distiller.log_weights_sparsity(self.model, epoch, [self.tflogger, self.pylogger])
                if self.args.masks_sparsity:
                    msglogger.info( \
                        distiller.masks_sparsity_tbl_summary( \
                            self.model, \
                            self.compression_scheduler))
            
            # ---------------------- validate_one_epoch ---------------------- #
            # loss, psnr_score, ssim_score = self.validate_one_epoch(epoch, verbose=True, is_last_epoch = is_last_epoch)
            with collectors_context(self.activations_collectors["valid"]) as collectors:
                # vloss, vpsnr, vssim = distiller.apputils.siren_utils.siren_train_val_test_utils.validate(self.val_loader, self.model, self.criterion, 
                """
                loss, psnr_score, ssim_score = \
                    distiller.apputils.siren_utils.siren_train_val_test_utils.validate( \
                        # self.val_loader, self.model, self.criterion, \
                        # inputs_val, target_val, total_samples_val, batch_size_val, self.model, self.criterion, \
                        # [self.pylogger], self.args, epoch, is_last_epoch = is_last_epoch, msglogger=msglogger)
                        inputs_val, target_val, self.model, self.criterion)
                """
                self.model.eval()

                # end = time.time()
                with torch.no_grad():
                    output, _ = self.model(inputs_val)
                    # objective_loss = criterion(output, target).item()
                    loss = self.criterion(output, target).item()
                    
                    sidelenght = output.size()[1]
                    
                    arr_output = output.cpu().view(sidelenght).detach().numpy()
                    arr_output = np.clip((arr_output / 2.) + 0.5, a_min=0., a_max=1.)

                    arr_gt = target_val.cpu().view(sidelenght).detach().numpy()
                    arr_gt = (arr_gt / 2.) + 0.5

                    # val_psnr = psnr(arr_gt, arr_output,data_range=1.)
                    # val_mssim = ssim(arr_gt, arr_output,data_range=1.)
                    psnr_score = psnr(arr_gt, arr_output,data_range=1.)
                    ssim_score = ssim(arr_gt, arr_output,data_range=1.)
                
                # Val: log activation/weigth-sparsity stats 
                # ------------------------------------------#
                distiller.log_activation_statistics(
                    epoch, \
                    "valid", \
                    loggers=[self.tflogger], \
                    collector=collectors["sparsity"])
                save_collectors_data(collectors, msglogger.logdir)

            # [On Epoch End] => Distiller framework operats by means of scheduler instance
            # as main orchestrator of the model's compression process.
            self.compression_scheduler.on_epoch_end( \
                epoch, \
                self.optimizer, \
                metrics={'min': loss,})

            # ---------------------- save/show scores (Mse, Psnr, Ssim) ---------------------- #
            """
            prune_details = _check_pruning_met_layers_sparse( \
                self.compression_scheduler, \
                self.model, epoch, \
                self.args, early_stopping_agp=self.early_stopping_agp, \
                save_mid_pr=self.save_mid_pr, \
                prune_details=prune_details)
            """
            if epoch >= 0 and epoch % self.args.print_freq == 0:
                # ---------------------- log train data ---------------------- #
                # msglogger.info('\n')
                msglogger.info('--- train (epoch=%d/%d)-----------', epoch, self.ending_epoch)
                # _log_training_progress(loggers=[self.tflogger, self.pylogger])
                stats_dict = OrderedDict()
                for loss_name, meter in losses.items(): stats_dict[loss_name] = meter.mean
                stats_dict['LR'] = self.optimizer.param_groups[0]['lr']
                stats_dict['Time'] = batch_time # batch_time.mean
                stats = ('Performance/Training/', stats_dict)

                params = self.model.named_parameters() if self.args.log_params_histograms else None
                distiller.log_training_progress(stats, params, # epoch, steps_completed,
                    epoch, 1, math.ceil(total_samples / batch_size),
                    self.args.print_freq, loggers=loggers)
                # _log_train_epoch_pruning(self.args, epoch, prune_details=prune_details)

                _, total = distiller.weights_sparsity_tbl_summary(self.model, return_total_sparsity=True)
                msglogger.info(f"Total Sparsity Achieved: {total}")
                
                # ---------------------- log val data ---------------------- #
                # msglogger.info('\n')
                msglogger.info('--- validation (epoch=%d/%d)-----------', epoch, self.ending_epoch)
                # _log_validation_progress(loggers=[self.tflogger, self.pylogger])
                """msglogger.info('==> MSE: %.7f   PSNR: %.7f   SSIM: %.7f\n', \
                    # losses['objective_loss'].mean, metrices['psnr'].mean(), metrices['ssim'].mean())
                    # losses['objective_loss'].mean, metrices['psnr'].mean, metrices['ssim'].mean)
                    loss, psnr_score, ssim_score)
                """
                stats = ('Performance/Validation/',
                    OrderedDict([('Loss', loss), # vloss
                    ('PSNR', psnr_score), # vpsnr
                    ('SSIM', ssim_score), # vssim
                ]))
                distiller.log_training_progress(stats, None, epoch, steps_completed=0,
                                            total_steps=1, log_freq=1, loggers=loggers)
            else:
                # _log_training_progress(); _log_validation_progress()
                # ---------------------- log train data ---------------------- #
                stats_dict = OrderedDict()
                for loss_name, meter in losses.items(): stats_dict[loss_name] = meter.mean
                stats_dict['LR'] = self.optimizer.param_groups[0]['lr']
                stats_dict['Time'] = batch_time # batch_time.mean
                stats = ('Performance/Training/', stats_dict)

                params = self.model.named_parameters() if self.args.log_params_histograms else None
                distiller.log_training_progress(stats, params, # epoch, steps_completed,
                    epoch, 1, math.ceil(total_samples / batch_size),
                    self.args.print_freq, loggers=loggers[0])
                
                # ---------------------- log val data ---------------------- #
                stats = ('Performance/Validation/',
                    OrderedDict([('Loss', loss), # vloss
                    ('PSNR', psnr_score), # vpsnr
                    ('SSIM', ssim_score), # vssim
                ]))
                distiller.log_training_progress(stats, None, epoch, steps_completed=0,
                                            total_steps=1, log_freq=1, loggers=loggers[0])
            
            # ---------------------- check whether to save middlle checkpoint data ---------------------- #
            # self._finalize_epoch(epoch, loss, psnr_score, ssim_score, is_last_epoch = False, prune_details=prune_details)
            self._finalize_epoch(
                epoch,
                loss, psnr_score, ssim_score,
                is_last_epoch=False, prune_details=prune_details)

            # Check whether to early halt training - when desired overall sparsity level is met.
            if self.early_stopping_agp is not None and self.early_stopping_agp.stop_training():
                msglogger.info("Early Stopping Halted training!")
                # self._finalize_epoch(epoch, loss, psnr_score, ssim_score, is_last_epoch = False, prune_details=prune_details)
                break
        
        # ---------------------- Save last cycle run data ---------------------- #
        # self._finalize_epoch(epoch, loss, psnr_score, ssim_score, is_last_epoch = True, prune_details=prune_details)
        old_name = str(self.args.name)
        self.args.name = "final_epoch"
        self._finalize_epoch(epoch, loss, psnr_score, ssim_score, is_last_epoch=True, prune_details=prune_details)
        self.args.name = old_name
        pass


    def run_plain_training_loop(self,):
        global msglogger

        total_samples = len(self.train_loader.sampler)
        batch_size = self.train_loader.batch_size

        inputs, target = next(iter(self.train_loader))
        inputs, target = inputs.to(self.args.device), target.to(self.args.device)

        total_samples_val = len(self.val_loader.sampler)
        batch_size_val = self.val_loader.batch_size

        inputs_val, target_val = next(iter(self.val_loader))
        inputs_val, target_val = inputs_val.to(self.args.device), target_val.to(self.args.device)

        for epoch in range(self.start_epoch, self.ending_epoch):
            is_last_epoch = epoch == self.ending_epoch - 1
            # ---------------------- train_one_epoch ---------------------- #
            # loss, psnr_score, ssim_score = self.train_validate_with_scheduling(epoch, is_last_epoch = is_last_epoch)
            with collectors_context(self.activations_collectors["train"]) as collectors:
                loss = \
                    distiller.apputils.siren_utils.siren_train_val_test_utils.train(
                        # self.train_loader, self.model,
                        inputs, target, total_samples, batch_size,
                        self.model,
                        self.criterion, self.optimizer, 
                        epoch, self.compression_scheduler,
                        loggers=[self.tflogger, self.pylogger], args=self.args, is_last_epoch = is_last_epoch,
                        early_stopping_agp=self.early_stopping_agp,
                        save_mid_pr=self.save_mid_pr, msglogger=msglogger)
            
            # ---------------------- validate_one_epoch ---------------------- #
            # loss, psnr_score, ssim_score = self.validate_one_epoch(epoch, verbose=True, is_last_epoch = is_last_epoch)
            with collectors_context(self.activations_collectors["valid"]) as collectors:
                # vloss, vpsnr, vssim = distiller.apputils.siren_utils.siren_train_val_test_utils.validate(self.val_loader, self.model, self.criterion, 
                loss, psnr_score, ssim_score = distiller.apputils.siren_utils.siren_train_val_test_utils.validate(
                    # self.val_loader, self.model, self.criterion, 
                    inputs_val, target_val, total_samples_val, batch_size_val, self.model, self.criterion, \
                    [self.pylogger], self.args, epoch, is_last_epoch = is_last_epoch, msglogger=msglogger)
                distiller.log_activation_statistics(epoch, "valid", loggers=[self.tflogger],
                                                    collector=collectors["sparsity"])
                save_collectors_data(collectors, msglogger.logdir)
            # if verbose:
            stats = ('Performance/Validation/',
            OrderedDict([('Loss', loss), # vloss
                ('PSNR', psnr_score), # vpsnr
                ('SSIM', ssim_score), # vssim
            ]))
            distiller.log_training_progress(stats, None, epoch, steps_completed=0,
                                            total_steps=1, log_freq=1, loggers=[self.tflogger])
            is_last_epoch = epoch == self.ending_epoch - 1
            self._finalize_epoch(epoch, loss, psnr_score, ssim_score, is_last_epoch = is_last_epoch)
        
        
    def run_training_loop(self):
        """Run the main training loop with compression.

        For each epoch:
            train_one_epoch
            validate_one_epoch
            finalize_epoch
        """
        if self.start_epoch >= self.ending_epoch:
            msglogger.error(
                'epoch count is too low, starting epoch is {} but total epochs set to {}'.format(
                self.start_epoch, self.ending_epoch))
            raise ValueError('Epochs parameter is too low. Nothing to do.')

        # Load the datasets lazily
        self.load_datasets()
        self.performance_tracker.reset()

        if self.compression_scheduler:
            msglogger.info("=> Running with scheduler")
            self.run_training_loop_with_scheduler()
            # self.run_plain_training_loop()
            return self.performance_tracker.perf_scores_history
        """
        for epoch in range(self.start_epoch, self.ending_epoch):
            is_last_epoch = epoch == self.ending_epoch - 1
            is_one_to_save_pruned = False
            if self.save_mid_pr is not None:
                is_one_to_save_pruned = self.save_mid_pr.is_one_to_save()
            if epoch >= 0 and epoch % self.args.print_freq == 0:
                msglogger.info('\n')
            loss, psnr_score, ssim_score = self.train_validate_with_scheduling(epoch, is_last_epoch = is_last_epoch)
            self._finalize_epoch(epoch, loss, psnr_score, ssim_score, is_last_epoch = is_last_epoch, is_one_to_save_pruned=is_one_to_save_pruned)

            if self.early_stopping_agp is not None and self.early_stopping_agp.stop_training():
                self._finalize_epoch(epoch, loss, psnr_score, ssim_score, is_last_epoch = True)
                break
        """

        # msglogger.info("=> Running without scheduler")
        # self.run_plain_training_loop()
        # return self.performance_tracker.perf_scores_history
        pass


    def validate(self, epoch=-1, is_last_epoch = False):
        # self.load_datasets()
        total_samples_val = len(self.train_loader.sampler)
        batch_size_val = self.train_loader.batch_size

        inputs_val, target_val = next(iter(self.train_loader))
        inputs_val, target_val = inputs_val.to(self.args.device), target_val.to(self.args.device)

        return distiller.apputils.siren_utils.siren_train_val_test_utils.validate(
            # self.val_loader,
            # inputs_val, target_val, total_samples_val, batch_size_val, \
            # self.model, self.criterion, \
            # [self.tflogger, self.pylogger], self.args, epoch, is_last_epoch = is_last_epoch, msglogger=msglogger)
            inputs_val, target_val, self.model, self.criterion)


    def test(self):
        global msglogger

        # self.test_mode_on = True
        self.load_datasets()
        start_time = time.time()
        msglogger.info('--- test ---------------------')
        result_test = test(self.test_loader, self.model, self.criterion,
                    self.pylogger, self.activations_collectors, args=self.args, test_mode_on = self.test_mode_on, msglogger=msglogger)
        # msglogger.info(f"Test Inference Time: {time.time() - start_time}")
        loss, psnr_score, ssim_score = result_test
        
        end_time = time.time() - start_time
        msglogger.info('==> MSE: %.7f   PSNR: %.7f   SSIM: %.7f   TIME: %.7f\n', \
                # losses['objective_loss'].mean, metrices['psnr'].mean(), metrices['ssim'].mean())
                # losses['objective_loss'].mean, metrices['psnr'].mean, metrices['ssim'].mean)
                loss, psnr_score, ssim_score, end_time)
        self.test_mode_on = False
        return loss, psnr_score, ssim_score, end_time


# ----------------------------------------------------------------------------------------------- #
# SirenRegressorCompressor: Util Functions
# ----------------------------------------------------------------------------------------------- #
def test(test_loader, model, criterion, loggers=None, activations_collectors=None, args=None, test_mode_on = True, msglogger = None):

    """Model Test.
    Return
    ------
    `losses` - list python object keeping MSE, PSNR and SSIM scores in that precise order.\n
    """
    if args is None:
        args = SirenRegressorCompressor.mock_args()
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)

    total_samples_val = len(test_loader.sampler)
    batch_size_val = test_loader.batch_size

    inputs_test, target_test = next(iter(test_loader))
    inputs_val, target_test = inputs_test.to(args.device), target_test.to(args.device)

    with collectors_context(activations_collectors["test"]) as collectors:
        # losses = distiller.apputils.siren_utils.siren_train_val_test_utils._validate(test_loader, model, criterion, loggers, args, test_mode_on = test_mode_on, msglogger=msglogger)
        losses = distiller.apputils.siren_utils.siren_train_val_test_utils.validate(
            # test_loader,
            # inputs_val, target_val, total_samples_val, batch_size_val, \
            # model, criterion, loggers, args, test_mode_on = test_mode_on, msglogger=msglogger)
            inputs_val, target_test, model, criterion)
        distiller.log_activation_statistics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    return losses


def init_regressor_compression_arg_parser(include_ptq_lapq_args=False):
    return distiller.apputils.siren_utils.siren_init_utils.init_regressor_compression_arg_parser(include_ptq_lapq_args=include_ptq_lapq_args)


def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    class missingdict(dict):
        """This is a little trick to prevent KeyError"""
        def __missing__(self, key):
            return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

    genCollectors = lambda: missingdict({
        "sparsity_ofm":      SummaryActivationStatsCollector(model, "sparsity_ofm",
            lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to disk.

    File type and format of contents are collector-specific.
    """
    for name, collector in collectors.items():
        msglogger.info('Saving data for collector {}...'.format(name))
        file_path = collector.save(os.path.join(directory, name))
        msglogger.info("Saved to {}".format(file_path))


def load_data(args, fixed_subset=False, sequential=False, load_train=True, load_val=True, load_test=True):
    """Load data for training/val/or test purposes.
    Args:
    -----
    `args` - Namespace compliant python object with details needed to determine how data will be loaded or which data will be loaded.\n
    `fixed_subset` - bool object which defaults to False and suggest whether subset of data will be fixed or not.\n
    `sequential` - bool object which default to False.\n
    `load_train` - bool object which defaults to True.\n
    `load_val` - bool object which defaults to True.\n
    `load_test` which defaults to True.\n
    """
    global msglogger

    test_only = not load_train and not load_val

    # pprint(args.data)
    # sys.exit(0)

    if args.data is None or len(args.data) == 0:
        msglogger.info("=> Loading 'Cameramen' Input Image as Target Data")
    else:
        image_name = os.path.basename(args.data)
        msglogger.info(f"=> Loading '{image_name}' Input Image as Target Data")
        pass

    train_loader, val_loader, test_loader, _ = distiller.apputils.load_data(args.dataset, args.arch,
                              os.path.expanduser(args.data), args.batch_size,
                              args.workers, args.validation_split, args.deterministic,
                              args.effective_train_size, args.effective_valid_size, args.effective_test_size,
                              fixed_subset, sequential, test_only)
    assert test_loader != None, "test_loader is None!"
    # print(test_loader)
    # print(len(test_loader.sampler))
    # sys.exit(0)
    if test_only:
        msglogger.info('Dataset sizes:\n\ttest=%d', len(test_loader.sampler))
    else:
        msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                       len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    loaders = (train_loader, val_loader, test_loader)
    flags = (load_train, load_val, load_test)
    loaders = [loaders[i] for i, flag in enumerate(flags) if flag]
    
    if len(loaders) == 1:
        # Unpack the list for convenience
        loaders = loaders[0]
    return loaders


# Temporary patch until we refactor early-exit handling
def _is_earlyexit(args):
    return hasattr(args, 'earlyexit_thresholds') and args.earlyexit_thresholds


def inception_training_loss(output, target, criterion, args):
    """Compute weighted loss for Inception networks as they have auxiliary classifiers

    Auxiliary classifiers were added to inception networks to tackle the vanishing gradient problem
    They apply softmax to outputs of one or more intermediate inception modules and compute auxiliary
    loss over same labels.
    Note that auxiliary loss is purely used for training purposes, as they are disabled during inference.

    GoogleNet has 2 auxiliary classifiers, hence two 3 outputs in total, output[0] is main classifier output,
    output[1] is aux2 classifier output and output[2] is aux1 classifier output and the weights of the
    aux losses are weighted by 0.3 according to the paper (C. Szegedy et al., "Going deeper with convolutions,"
    2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, 2015, pp. 1-9.)

    All other versions of Inception networks have only one auxiliary classifier, and the auxiliary loss
    is weighted by 0.4 according to PyTorch documentation
    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
    """
    weighted_loss = 0
    if args.arch == 'googlenet':
        # DEFAULT, aux classifiers are NOT included in PyTorch Pretrained googlenet model as they are NOT trained,
        # they are only present if network is trained from scratch. If you need to fine tune googlenet (e.g. after
        # pruning a pretrained model), then you have to explicitly enable aux classifiers when creating the model
        # DEFAULT, in case of pretrained model, output length is 1, so loss will be calculated in main training loop
        # instead of here, as we enter this function only if output is a tuple (len>1)
        # TODO: Enable user to feed some input to add aux classifiers for pretrained googlenet model
        outputs, aux2_outputs, aux1_outputs = output    # extract all 3 outputs
        loss0 = criterion(outputs, target)
        loss1 = criterion(aux1_outputs, target)
        loss2 = criterion(aux2_outputs, target)
        weighted_loss = loss0 + 0.3*loss1 + 0.3*loss2
    else:
        outputs, aux_outputs = output    # extract two outputs
        loss0 = criterion(outputs, target)
        loss1 = criterion(aux_outputs, target)
        weighted_loss = loss0 + 0.4*loss1
    return weighted_loss


def earlyexit_validate_loss(output, target, criterion, args):
    # We need to go through each sample in the batch itself - in other words, we are
    # not doing batch processing for exit criteria - we do this as though it were batch size of 1,
    # but with a grouping of samples equal to the batch size.
    # Note that final group might not be a full batch - so determine actual size.
    this_batch_size = target.size(0)
    earlyexit_validate_criterion = nn.CrossEntropyLoss(reduce=False).to(args.device)

    for exitnum in range(args.num_exits):
        # calculate losses at each sample separately in the minibatch.
        args.loss_exits[exitnum] = earlyexit_validate_criterion(output[exitnum], target)
        # for batch_size > 1, we need to reduce this down to an average over the batch
        args.losses_exits[exitnum].add(torch.mean(args.loss_exits[exitnum]).cpu())

    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(args.num_exits - 1):
            if args.loss_exits[exitnum][batch_index] < args.earlyexit_thresholds[exitnum]:
                # take the results from early exit since lower than threshold
                args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                             torch.full([1], target[batch_index], dtype=torch.long))
                args.exit_taken[exitnum] += 1
                earlyexit_taken = True
                break                    # since exit was taken, do not affect the stats of subsequent exits
        # this sample does not exit early and therefore continues until final exit
        if not earlyexit_taken:
            exitnum = args.num_exits - 1
            args.exiterrors[exitnum].add(torch.tensor(np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)),
                                         torch.full([1], target[batch_index], dtype=torch.long))
            args.exit_taken[exitnum] += 1


def _convert_ptq_to_pytorch(model, args):
    msglogger.info('Converting Distiller PTQ model to PyTorch quantization API')
    dummy_input = distiller.get_dummy_input(input_shape=model.input_shape)
    model = distiller.quantization.convert_distiller_ptq_model_to_pytorch(model, dummy_input, backend=args.qe_pytorch_backend)
    msglogger.debug('\nModel after conversion:\n{}'.format(model))
    args.device = 'cpu'
    return model


def evaluate_model(test_loader, model, criterion, loggers, activations_collectors=None, args=None, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate
    global msglogger
    if not isinstance(loggers, list):
        loggers = [loggers]

    if not args.quantize_eval:
        # Handle case where a post-train quantized model was loaded, and user wants to convert it to PyTorch
        if args.qe_convert_pytorch:
            model = _convert_ptq_to_pytorch(model, args)
        return test(test_loader, model, criterion, loggers, activations_collectors, args=args, msglogger=msglogger)
    else:
        return quantize_and_test_model(test_loader, model, criterion, args, loggers,
                                       scheduler=scheduler, save_flag=True)


def quantize_and_test_model(test_loader, model, criterion, args, loggers=None, scheduler=None, save_flag=True):
    """Collect stats using test_loader (when stats file is absent),

    clone the model and quantize the clone, and finally, test it.
    args.device is allowed to differ from the model's device.
    When args.qe_calibration is set to None, uses 0.05 instead.

    scheduler - pass scheduler to store it in checkpoint
    save_flag - defaults to save both quantization statistics and checkpoint.
    """
    global msglogger

    if hasattr(model, 'quantizer_metadata') and \
            model.quantizer_metadata['type'] == distiller.quantization.PostTrainLinearQuantizer:
        raise RuntimeError('Trying to invoke post-training quantization on a model that has already been post-'
                           'train quantized. Model was likely loaded from a checkpoint. Please run again without '
                           'passing the --quantize-eval flag')
    if not (args.qe_dynamic or args.qe_stats_file or args.qe_config_file):
        args_copy = copy.deepcopy(args)
        args_copy.qe_calibration = args.qe_calibration if args.qe_calibration is not None else 0.05

        # set stats into args stats field
        args.qe_stats_file = acts_quant_stats_collection(
            model, criterion, loggers, args_copy, save_to_file=save_flag)

    args_qe = copy.deepcopy(args)
    if args.device == 'cpu':
        # NOTE: Even though args.device is CPU, we allow here that model is not in CPU.
        qe_model = distiller.make_non_parallel_copy(model).cpu()
    else:
        qe_model = copy.deepcopy(model).to(args.device)

    quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(qe_model, args_qe)
    dummy_input = distiller.get_dummy_input(input_shape=model.input_shape)
    quantizer.prepare_model(dummy_input)

    if args.qe_convert_pytorch:
        qe_model = _convert_ptq_to_pytorch(qe_model, args_qe)

    test_res = distiller.apputils.siren_utils.siren_train_val_test_utils.test(test_loader, qe_model, criterion, loggers, args=args_qe, msglogger=msglogger)

    if save_flag:
        checkpoint_name = 'quantized'
        distiller.apputils.save_checkpoint(0, args_qe.arch, qe_model, scheduler=scheduler,
            name='_'.join([args_qe.name, checkpoint_name]) if args_qe.name else checkpoint_name,
            dir=msglogger.logdir, extras={'quantized_mse': test_res[0]})

    del qe_model
    return test_res


def acts_quant_stats_collection(model, criterion, loggers, args, test_loader=None, save_to_file=False):
    msglogger.info('Collecting quantization calibration stats based on {:.1%} of test dataset'
                   .format(args.qe_calibration))
    if test_loader is None:
        tmp_args = copy.deepcopy(args)
        tmp_args.effective_test_size = tmp_args.qe_calibration
        # Batch size 256 causes out-of-memory errors on some models (due to extra space taken by
        # stats calculations). Limiting to 128 for now.
        # TODO: Come up with "smarter" limitation?
        tmp_args.batch_size = min(128, tmp_args.batch_size)
        test_loader = load_data(tmp_args, fixed_subset=True, load_train=False, load_val=False)
    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    with distiller.get_nonparallel_clone_model(model) as cmodel:
        return collect_quant_stats(cmodel, test_fn, classes=None,
                                   inplace_runtime_check=True, disable_inplace_attrs=True,
                                   save_dir=msglogger.logdir if save_to_file else None)


def acts_histogram_collection(model, criterion, loggers, args):
    msglogger.info('Collecting activation histograms based on {:.1%} of test dataset'
                   .format(args.activation_histograms))
    model = distiller.utils.make_non_parallel_copy(model)
    args.effective_test_size = args.activation_histograms
    test_loader = load_data(args, fixed_subset=True, load_train=False, load_val=False)
    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    collect_histograms(model, test_fn, save_dir=msglogger.logdir,
                       classes=None, nbins=2048, save_hist_imgs=True)


def _log_best_scores(performance_tracker, logger, how_many=-1):
    """Utility to log the best scores.

    This function is currently written for pruning use-cases, but can be generalized.
    """
    try:
        assert isinstance(performance_tracker, (distiller.apputils.SparsityMSETracker))
    except:
        assert isinstance(performance_tracker, (SparsityMSETracker))
    if how_many < 1:
        how_many = performance_tracker.max_len
    how_many = min(how_many, performance_tracker.max_len)
    best_scores = performance_tracker.best_scores(how_many)
    for score in best_scores:
        logger.info('==> Best [MSE: %.7f   PSNR: %.7f   SSIM: %.7f   Sparsity:%.2f   NNZ-Params: %d on epoch: %d]',
                    score.mse, score.psnr_score, score.ssim_score, score.sparsity, -score.params_nnz_cnt, score.epoch)


# ----------------------------------------------------------------------------------------------- #
# Under-test functions
# ----------------------------------------------------------------------------------------------- #
def _log_train_epoch_pruning(args, epoch, prune_details = {}):
    """Log to json file information and data about when pruning take places per layer."""
    # global msglogger
    # global PRUNE_DETAILS
    global msglogger

    if prune_details == {}: return

    out_file_data = os.path.join(f'{msglogger.logdir}', 'data.json')
    str_data = json.dumps(prune_details)

    msglogger.info(f"--- dump pruning data (epoch={epoch}) ---------")
    msglogger.info(f"Data saved to: {out_file_data}")
    # msglogger.info(str_data)
    try:
        with open(out_file_data, 'w') as outfile:
            json.dump(prune_details, outfile)
    except Exception as err:
        msglogger.info(f"{str(err)}.\nError occour when attempting to saving: {out_file_data}")


def _check_pruning_met_layers_sparse(compression_scheduler, model, epoch, args, early_stopping_agp = None, save_mid_pr = None, prune_details={}):
    """Update dictionary storing data and information about when pruning takes places for each layer."""
    # global msglogger
    # global PRUNE_DETAILS
    global TOLL
    global msglogger

    _, total, df = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True, return_df=True)

    if early_stopping_agp:
        early_stopping_agp.check_total_sparsity_is_met(curr_sparsity=total)
        # is_triggered = early_stopping_agp.is_triggered()
        # if is_triggered:
        # epochs_done, total_epochs_to_patience = early_stopping_agp.update_trail_epochs()
        # msglogger.info(f"EarlyStoppingAGP: is_triggered={is_triggered} - before halting training: ({epochs_done}/{total_epochs_to_patience})")
        if early_stopping_agp.is_triggered_once():
            msglogger.info(f"(EarlyStoppingAGP) Total sparsity: {total} has been met at epoch: {epoch}")
    if save_mid_pr:
        if save_mid_pr.is_rate_into_middle_prune_rates(a_prune_rate=total, epoch=epoch):
            msglogger.info(f"(SaveMiddlePruneRate) Mid sparsity: {total} has been met at epoch: {epoch}")
    
    policies_list = list(compression_scheduler.sched_metadata.keys())
    if policies_list == []: return
    
    for policy in policies_list:
        # sched_metadata = compression_scheduler.sched_metadata[policy]
        if not hasattr(policy, 'pruner') : continue
        pruner = policy.pruner
        if isinstance(pruner, AutomatedGradualPruner):  
            final_sparsity = pruner.agp_pr.final_sparsity
            for param_name in pruner.params_names:
                data_tmp = df[df["Name"] == param_name].values[0]
                data_tmp_dict = dict(zip(list(df.columns), data_tmp))
                if param_name not in prune_details.keys():
                    # Check and eventually Insert new layer
                    pruner_name = str(pruner).split(" ")[0].split(".")[-1]
                    keys = "epoch,param_name,pruner,Fine (%),satisfyed,toll".split(",")
                    record_data = [epoch, param_name, pruner_name, data_tmp_dict["Fine (%)"], 0, TOLL]
                    prune_details[param_name] = dict(zip(keys, record_data))
                elif data_tmp_dict["Fine (%)"] >= (final_sparsity * 100 - TOLL) or data_tmp_dict["Fine (%)"] >= final_sparsity * 100:
                    if float(prune_details[param_name]["Fine (%)"]) < data_tmp_dict["Fine (%)"]:
                        # Update if necessary insert new layer
                        pruner_name = str(pruner).split(" ")[0].split(".")[-1]
                        keys = "epoch,param_name,pruner,Fine (%),satisfyed,toll".split(",")
                        record_data = [epoch, param_name, pruner_name, data_tmp_dict["Fine (%)"], 1, TOLL]
                        prune_details[param_name] = dict(zip(keys, record_data))
                else:
                    pruner_name = str(pruner).split(" ")[0].split(".")[-1]
                    keys = "epoch,param_name,pruner,Fine (%),satisfyed,toll".split(",")
                    record_data = [epoch, param_name, pruner_name, data_tmp_dict["Fine (%)"], 0, TOLL]
                    prune_details[param_name] = dict(zip(keys, record_data))
    return prune_details


def save_predicted_data(test_loader, model, criterion, loggers, activations_collectors=None, args=None, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate

    global msglogger

    if not isinstance(loggers, list):
        loggers = [loggers]

    if not args.quantize_eval:
        # Handle case where a post-train quantized model was loaded, and user wants to convert it to PyTorch
        if args.qe_convert_pytorch:
            model = _convert_ptq_to_pytorch(model, args)
    else:
        quantize_and_test_model(test_loader, model, criterion, args, loggers,scheduler=scheduler, save_flag=True)
    return predict_image(test_loader, model, criterion, loggers, activations_collectors, args=args, msglogger=msglogger)


def predict_image(test_loader, model, criterion, loggers=None, activations_collectors=None, args=None):
    """Model Test"""
    msglogger.info('--- predict data ---------------------')
    if args is None:
        args = SirenRegressorCompressor.mock_args()
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)

    with collectors_context(activations_collectors["test"]) as collectors:
        lossses = _save_predicted_image(test_loader, model, criterion, loggers, args)
        distiller.log_activation_statistics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    return lossses


def _save_predicted_image(data_loader, model, criterion, loggers, args, epoch=-1, is_last_epoch=-1, msglogger = None):
    def _log_validation_progress():
        stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),])
        # if not _is_earlyexit(args): stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),])
        # else:
        """stats_dict = OrderedDict()
        for exitnum in range(args.num_exits):
            la_string = 'LossAvg' + str(exitnum)
            stats_dict[la_string] = args.losses_exits[exitnum].mean
        """
        stats = ('Performance/Validation/', stats_dict)
        distiller.log_training_progress(stats, None, epoch, steps_completed,
                                        total_steps, args.print_freq, loggers)

    """Execute the validation/test loop for saving image."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    metrices = {'ssim': tnt.AverageValueMeter(), 'psnr': tnt.AverageValueMeter()}
    # metrices = { 'psnr': [], 'ssim': [] }

    """
    if _is_earlyexit(args):
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits
    """

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size

    total_steps = total_samples / batch_size
    # if epoch >= 0 and epoch % args.print_freq == 0: msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    with torch.no_grad():

        validation_step = 0
        inputs, target = next(iter(data_loader))
        # for validation_step, (inputs, target) in enumerate(data_loader):
        inputs, target = inputs.to(args.device), target.to(args.device)
        # compute output from model
        output, _ = model(inputs)

        predicted_image_path = os.path.join(args.output_dir, 'predicted_image.txt')
        sidelenght = output.size()[1]
        arr_image = output.cpu().view(sidelenght).detach().numpy()
        np.savetxt(predicted_image_path, arr_image)

        # if not _is_earlyexit(args):
        # compute loss
        loss = criterion(output, target)
        # measure accuracy and record loss
        losses['objective_loss'].add(loss.item())
        val_psnr, val_mssim = compute_desired_metrices(
            model_output = output, gt = target, data_range=1.)
        # metrices['psnr'].append(val_psnr); metrices['ssim'].append(val_mssim)
        metrices['psnr'].add(val_psnr); metrices['ssim'].add(val_mssim)
        
        # else: earlyexit_validate_loss(output, target, criterion, args)

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

        steps_completed = (validation_step+1)
        # if steps_completed > args.print_freq and steps_completed % args.print_freq == 0:
        if epoch >= 0 and epoch % args.print_freq == 0:
            _log_validation_progress()

    # if args.wandb_logging: wandb.log({"loss": losses['objective_loss'].mean, 'psnr': losses['psnr'].mean, 'ssim': metrices['ssim'].mea})
    # if not _is_earlyexit(args):
    # metrices['psnr'] = np.array(metrices['psnr']); metrices['ssim'] = np.array(metrices['ssim'])
    if (epoch >= 0 and epoch % args.print_freq == 0) or is_last_epoch:
        msglogger.info('==> MSE: %.7f   PSNR: %.7f   SSIM: %.7f\n', \
            # losses['objective_loss'].mean, metrices['psnr'].mean(), metrices['ssim'].mean())
            losses['objective_loss'].mean, metrices['psnr'].mean, metrices['ssim'].mean)
    elif test_mode_on:
        # if args.evaluate and test_mode_on:
        msglogger.info('==> MSE: %.7f   PSNR: %.7f   SSIM: %.7f\n', \
            # losses['objective_loss'].mean, metrices['psnr'].mean(), metrices['ssim'].mean())
            losses['objective_loss'].mean, metrices['psnr'].mean, metrices['ssim'].mean)
        # return losses['objective_loss'].mean, metrices['psnr'].mean(), metrices['ssim'].mean()
    # else:
    #    losses_exits_stats = earlyexit_validate_stats(args)
    #     return losses_exits_stats[args.num_exits-1]
    return losses['objective_loss'].mean, metrices['psnr'].mean, metrices['ssim'].mean

# ----------------------------------------------------------------------------------------------- #
# Under-test Classes
# ----------------------------------------------------------------------------------------------- #
class EarlyStoppingAGP(object):
    """Class defining EarlyStoppingAGP behaviour."""
    def __init__(self, target_sparsity, toll=2.0, patience=5, trail_epochs=5):
        """Instantiate EarlyStoppingAGP object.
        Args:
        -----
        `target_sparsity` - float, rate of sparsity level to achieve.\n
        `toll` - float, tolerance for allowing early-stopping to be triggered even if total target sparsity is not met, in order to shorten training time.\n
        `patience` - number of epoch before triggering EarlyStoppingAGP when toll is met.\n
        `trail_epochs` - number of epoch after which train will be stopped, once early stopping have been triggered.\n
        """
        self.target_sparsity = target_sparsity
        self.toll = toll
        self.patience = patience
        self.trail_epochs = trail_epochs

        self._steps_to_trail_epochs = 0
        self._steps_to_patience = 0
        
        self._is_triggered_flag = False
        self._curr_sparsity = 0.0

    def is_triggered(self,):
        """Check whether is trieggered.
        Return:
        -------
        `bool` - illustrating whether or not earlystopping-agp have been triggered.\n
        """
        return self._is_triggered_flag
    def is_triggered_once(self,):
        """Check whether is trieggered.
        Return:
        -------
        `bool` - illustrating whether or not earlystopping-agp have been triggered.\n
        """
        if self._is_triggered_flag:
            return False
        return self._is_triggered_flag
    
    def check_total_sparsity_is_met(self, curr_sparsity):
        """Check whether total sparsity is met.
        Args:
        -----
        `curr_sparsity` - float, current sparsity level.\n
        """
        
        if curr_sparsity >= self.target_sparsity:
            self._is_triggered_flag = True
        elif curr_sparsity >= self.target_sparsity - self.toll:
            if self._curr_sparsity == curr_sparsity:
                self._steps_to_patience += 1
            else:
                self._curr_sparsity = curr_sparsity
            if self._steps_to_patience == self.patience:
                self._is_triggered_flag = True
        else:
            self._curr_sparsity = curr_sparsity

    def update_trail_epochs(self,):
        """Update info about patience to let pass before stopping defenitely training.
        Return:
        -------
        `int` - remaining epochs before training will be halted.\n
        `int` - total epoch to patience.\n
        """
        self._steps_to_trail_epochs += 1
        return self._steps_to_trail_epochs, self.trail_epochs

    def stop_training(self):
        """Check wheter it is necessary to stop training.
        Return:
        -------
        `bool` - illustrating whether or not it is mandatory to stop training.\n
        """
        if self.is_triggered():
            if self._steps_to_trail_epochs == self.trail_epochs:
                return True
        return False


class SaveMiddlePruneRate(object):
    """Class containing behaviour for tracking mid prune targets to save a part."""
    def __init__(self, middle_prune_rates):
        """Instantiate object to track mid prune rates achieved while pruning a model.
        Args
        ----
        `middle_prune_rates` - list of float values to be checked as prune rate reached.\n
        """
        global msglogger
        self.msglogger = msglogger
        self.middle_prune_rates = \
            middle_prune_rates if isinstance(middle_prune_rates, list) \
            else list(middle_prune_rates)
        self.middle_prune_rates = iter(self.middle_prune_rates)
        self.found_middle_prune_rates = False
        self.prune_rate_val = -1
        self.curr_val = next(self.middle_prune_rates)

    def is_rate_into_middle_prune_rates(self, a_prune_rate, epoch=-1):
        is_found = False
        """Check if new target prune rate  has been reached.
        Args
        ----
        `a_prune_rate` - float value to be checked as prune rate reached.\n
        Return
        ------
        `found` - bool, indicating if a new prune rate has been reached.\n
        """
        if a_prune_rate >= self.curr_val:
            while self.curr_val:
                if a_prune_rate < self.curr_val:
                    # self.curr_val = next(self.middle_prune_rates, None)
                    self.prune_rate_val = a_prune_rate
                    self.found_middle_prune_rates = True
                    self.msglogger.info(f"Found new intermediate Prune rate achieved: prune_rate={a_prune_rate}, epoch={epoch}")
                    return True
                self.curr_val = next(self.middle_prune_rates, None)
        return False
    def is_one_to_save(self,):
        """Check wheter there is one to save.
        Return
        ------
        `bool` - indicating if a new prune rate has been reached and must be saved.\n
        """
        return self.found_middle_prune_rates
    def get_rate_to_save(self,):
        """Get last rate achieved and not yet saved a part.
        Return
        ------
        `prune_rate_val` - float, indicating if a new prune rate has been reached and must be saved.\n
        """
        if self.found_middle_prune_rates:
            self.found_middle_prune_rates = False
            return self.prune_rate_val
        return None
