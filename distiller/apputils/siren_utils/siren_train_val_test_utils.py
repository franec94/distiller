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

TOLL = 2
PRUNE_DETAILS = dict()


# ----------------------------------------------------------------------------------------------- #
# Utils Section
# ----------------------------------------------------------------------------------------------- #

def early_exit_mode(args):
    return hasattr(args, 'earlyexit_lossweights') and args.earlyexit_lossweights


def earlyexit_loss(output, target, criterion, args):
    """Compute the weighted sum of the exits losses

    Note that the last exit is the original exit of the model (i.e. the
    exit that traverses the entire network.
    """
    weighted_loss = 0
    sum_lossweights = sum(args.earlyexit_lossweights)
    assert sum_lossweights < 1
    for exitnum in range(args.num_exits-1):
        if output[exitnum] is None:
            continue
        exit_loss = criterion(output[exitnum], target)
        weighted_loss += args.earlyexit_lossweights[exitnum] * exit_loss
        args.exiterrors[exitnum].add(output[exitnum].detach(), target)
    # handle final exit
    weighted_loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits-1], target)
    args.exiterrors[args.num_exits-1].add(output[args.num_exits-1].detach(), target)
    return weighted_loss


def _check_pruning_met_layers_sparse(compression_scheduler, model, epoch, args, early_stopping_agp = None, save_mid_pr = None, msglogger = None):
    """Update dictionary storing data and information about when pruning takes places for each layer."""
    # global msglogger
    global PRUNE_DETAILS
    global TOLL
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
        save_mid_pr.is_rate_into_middle_prune_rates(a_prune_rate=total, epoch=epoch)
    
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
                if data_tmp_dict["Fine (%)"] >= (final_sparsity * 100 - TOLL) or data_tmp_dict["Fine (%)"] >= final_sparsity * 100:
                    if param_name not in PRUNE_DETAILS.keys():
                        # Check and eventually Insert new layer
                        pruner_name = str(pruner).split(" ")[0].split(".")[-1]
                        keys = "epoch,param_name,pruner,Fine (%),satisfyed,toll".split(",")
                        record_data = [epoch, param_name, pruner_name, data_tmp_dict["Fine (%)"], 0, TOLL]
                        PRUNE_DETAILS[param_name] = dict(zip(keys, record_data))
                    elif float(PRUNE_DETAILS[param_name]["Fine (%)"]) < data_tmp_dict["Fine (%)"]:
                        # Update if necessary insert new layer
                        keys = "epoch,param_name,pruner,Fine (%),satisfyed,toll".split(",")
                        record_data = [epoch, param_name, pruner_name, data_tmp_dict["Fine (%)"], 1, TOLL]
                        PRUNE_DETAILS[param_name] = dict(zip(keys, record_data))


def _log_train_epoch_pruning(args, epoch, msglogger):
    """Log to json file information and data about when pruning take places per layer."""
    global FIND_EPOCH_FOR_PRUNING
    # global msglogger

    if PRUNE_DETAILS == {}: return

    out_file_data = os.path.join(f'{msglogger.logdir}', 'data.json')
    str_data = json.dumps(PRUNE_DETAILS)

    msglogger.info(f"--- dump pruning data (epoch={epoch}) ---------")
    msglogger.info(f"Data saved to: {out_file_data}")
    msglogger.info(str_data)
    try:
        with open(out_file_data, 'w') as outfile:
            json.dump(FIND_EPOCH_FOR_PRUNING, outfile)
    except Exception as err:
        msglogger.info(f"{str(err)}.\nError occour when attempting to saving: {out_file_data}")

def get_prune_detail(): return PRUNE_DETAILS

def earlyexit_validate_stats(args, msglogger):
    # Print some interesting summary stats for number of data points that could exit early
    losses_exits_stats = [0] * args.num_exits
    sum_exit_stats = 0
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            sum_exit_stats += args.exit_taken[exitnum]
            msglogger.info("Exit %d: %d", exitnum, args.exit_taken[exitnum])
            losses_exits_stats[exitnum] += args.losses_exits[exitnum].mean
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            msglogger.info("Percent Early Exit %d: %.3f", exitnum,
                           (args.exit_taken[exitnum]*100.0) / sum_exit_stats)
    return losses_exits_stats
# ----------------------------------------------------------------------------------------------- #
# Train Section
# ----------------------------------------------------------------------------------------------- #
def train_via_scheduler(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args, is_last_epoch = False, early_stopping_agp=None, save_mid_pr=None, msglogger = None):
    """Training-with-compression loop for one epoch.
    
    For each training step in epoch:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)
    """
    # global msglogger

    if epoch >= 0 and epoch % args.print_freq == 0:
            msglogger.info('\n')
            msglogger.info('--- train (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- train ---------------------')

    def _log_training_progress():
        # Log some statistics

        # _, _, df = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True, return_df=True)
        stats_dict = OrderedDict()
        for loss_name, meter in losses.items():
            stats_dict[loss_name] = meter.mean
        stats_dict['LR'] = optimizer.param_groups[0]['lr']
        stats_dict['Time'] = batch_time.mean
        stats = ('Performance/Training/', stats_dict)

        params = model.named_parameters() if args.log_params_histograms else None
        distiller.log_training_progress(stats,
                                        params,
                                        epoch, steps_completed,
                                        steps_per_epoch, args.print_freq,
                                        loggers)

    OVERALL_LOSS_KEY = 'Overall Loss'
    OBJECTIVE_LOSS_KEY = 'Objective Loss'

    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    # For Early Exit, we define statistics for each exit, so
    # `exiterrors` is analogous to `classerr` in the non-Early Exit case
    if early_exit_mode(args): args.exiterrors = []

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    # if epoch >= 0 and epoch % args.print_freq == 0: msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to train mode
    model.train()
    end = time.time()
    inputs, target = next(iter(train_loader))
    train_step = 0
    # for train_step, (inputs, target) in enumerate(train_loader):
    # Measure data loading time
    data_time.add(time.time() - end)
    inputs, target = inputs.cuda(), target.cuda()

    # Execute the forward phase, compute the output and measure loss
    compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

    # if not hasattr(args, 'kd_policy') or args.kd_policy is None: output, _ = model(inputs)
    # else: output, _ = args.kd_policy.forward(inputs)
    output, _ = model(inputs)

    if not early_exit_mode(args): loss = criterion(output, target)
    else: loss = earlyexit_loss(output, target, criterion, args)
    
    # Record loss
    losses[OBJECTIVE_LOSS_KEY].add(loss.item())

    # Before running the backward phase, we allow the scheduler to modify the loss
    # (e.g. add regularization loss)
    agg_loss = \
        compression_scheduler.before_backward_pass(
            epoch,
            train_step, steps_per_epoch,
            loss,
            optimizer=optimizer, return_loss_components=True)
    loss = agg_loss.overall_loss
    losses[OVERALL_LOSS_KEY].add(loss.item())

    for lc in agg_loss.loss_components:
        if lc.name not in losses:
            losses[lc.name] = tnt.AverageValueMeter()
        losses[lc.name].add(lc.value.item())

    # Compute the gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    
    compression_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)
    optimizer.step()
    compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

    # measure elapsed time
    batch_time.add(time.time() - end)
    steps_completed = (train_step+1)

    # if steps_completed > args.print_freq and steps_completed % args.print_freq == 0:
    _check_pruning_met_layers_sparse(compression_scheduler, model, epoch, args, early_stopping_agp=early_stopping_agp, save_mid_pr=save_mid_pr, msglogger=msglogger)
    if epoch >= 0 and epoch % args.print_freq == 0 or is_last_epoch:
        _log_training_progress()
        _log_train_epoch_pruning(args, epoch, msglogger)
    end = time.time()
    #return acc_stats
    # NOTE: this breaks previous behavior, which returned a history of (top1, top5) values
    return losses[OVERALL_LOSS_KEY]


def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args, is_last_epoch = False, early_stopping_agp=None, save_mid_pr=None, msglogger = None):
    """Training-with-compression loop for one epoch.
    
    For each training step in epoch:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)
    """
    # global msglogger

    if epoch >= 0 and epoch % args.print_freq == 0:
            msglogger.info('\n')
            msglogger.info('--- train (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- train ---------------------')

    def _log_training_progress():
        # Log some statistics

        # _, _, df = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True, return_df=True)
        stats_dict = OrderedDict()
        for loss_name, meter in losses.items():
            stats_dict[loss_name] = meter.mean
        stats_dict['LR'] = optimizer.param_groups[0]['lr']
        stats_dict['Time'] = batch_time.mean
        stats = ('Performance/Training/', stats_dict)

        params = model.named_parameters() if args.log_params_histograms else None
        distiller.log_training_progress(stats,
                                        params,
                                        epoch, steps_completed,
                                        steps_per_epoch, args.print_freq,
                                        loggers)

    OVERALL_LOSS_KEY = 'Overall Loss'
    OBJECTIVE_LOSS_KEY = 'Objective Loss'

    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    # For Early Exit, we define statistics for each exit, so
    # `exiterrors` is analogous to `classerr` in the non-Early Exit case
    # if early_exit_mode(args): args.exiterrors = []

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    # if epoch >= 0 and epoch % args.print_freq == 0: msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to train mode
    model.train()
    end = time.time()
    inputs, target = next(iter(train_loader))
    # for train_step, (inputs, target) in enumerate(train_loader):
    # Measure data loading time
    data_time.add(time.time() - end)
    inputs, target = inputs.to(args.device), target.to(args.device)
    
    if not hasattr(args, 'kd_policy') or args.kd_policy is None:
        output, _ = model(inputs)
    else:
        output, _ = args.kd_policy.forward(inputs)

    # if not early_exit_mode(args): loss = criterion(output, target)
    # else: loss = earlyexit_loss(output, target, criterion, args)
    loss = criterion(output, target)
    
    # Record loss
    losses[OBJECTIVE_LOSS_KEY].add(loss.item())
    losses[OVERALL_LOSS_KEY].add(loss.item())

    # Compute the gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.add(time.time() - end)
    steps_completed = (train_step+1)

    # if steps_completed > args.print_freq and steps_completed % args.print_freq == 0:
    # _check_pruning_met_layers_sparse(compression_scheduler, model, epoch, args, early_stopping_agp=early_stopping_agp, save_mid_pr=save_mid_pr)
    if epoch >= 0 and epoch % args.print_freq == 0 or is_last_epoch:
        _log_training_progress()
        _log_train_epoch_pruning(args, epoch)
    end = time.time()
    #return acc_stats
    # NOTE: this breaks previous behavior, which returned a history of (top1, top5) values
    return losses[OVERALL_LOSS_KEY]


# ----------------------------------------------------------------------------------------------- #
# Test or Validate Section
# ----------------------------------------------------------------------------------------------- #
def validate(val_loader, model, criterion, loggers, args, epoch=-1, is_last_epoch = False, msglogger = None):
    """Model validation"""
    if epoch >= 0 and epoch % args.print_freq == 0 or is_last_epoch: msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else: msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch, is_last_epoch = is_last_epoch)


def test(test_loader, model, criterion, loggers=None, activations_collectors=None, args=None, test_mode_on = True, msglogger = None):

    """Model Test.
    Return
    ------
    `losses` - list python object keeping MSE, PSNR and SSIM scores in that precise order.\n
    """
    msglogger.info('--- test ---------------------')
    if args is None:
        args = SirenRegressorCompressor.mock_args()
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)

    with collectors_context(activations_collectors["test"]) as collectors:
        losses = _validate(test_loader, model, criterion, loggers, args, test_mode_on = test_mode_on)
        distiller.log_activation_statistics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    return losses


def _validate(data_loader, model, criterion, loggers, args, epoch=-1, test_mode_on = False, is_last_epoch = False, msglogger = None):
    """Validate model on validation set or test set, depending on which time instant it is called.
    Return
    ------
    `loss_score` - float value corresponding to MSE score.\n
    `psnr_score` - float value corresponding to PSNR score.\n
    `ssim_score` - float value corresponding to SSIM score.\n
    """
    global TARGET_TOTAL_SPARSITY

    def _log_validation_progress():
        stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),])
        #if not _is_earlyexit(args): # stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),])
        # else:
        """ stats_dict = OrderedDict()
        for exitnum in range(args.num_exits):
            la_string = 'LossAvg' + str(exitnum)
            stats_dict[la_string] = args.losses_exits[exitnum].mean
        """
        stats = ('Performance/Validation/', stats_dict)
        distiller.log_training_progress(stats, None, epoch, steps_completed,
                                        total_steps, args.print_freq, loggers)

    """Execute the validation/test loop."""
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
        # for validation_step, (inputs, target) in enumerate(data_loader):
        validation_step = 0
        inputs, target = next(iter(data_loader))
        
        inputs, target = inputs.to(args.device), target.to(args.device)        
        # compute output from model
        output, _ = model(inputs)

        # if not _is_earlyexit(args):
        # compute loss
        loss = criterion(output, target)
        # measure accuracy and record loss
        losses['objective_loss'].add(loss.item())
        # val_psnr, val_mssim = compute_desired_metrices(model_output = output, gt = target, data_range=1.)
        sidelenght = output.size()[1]

        arr_gt = target.cpu().view(sidelenght).detach().numpy()
        arr_gt = (arr_gt / 2.) + 0.5

        arr_output = output.cpu().view(sidelenght).detach().numpy()
        arr_output = (arr_output / 2.) + 0.5
        arr_output = np.clip(arr_output, a_min=0., a_max=1.)

        val_psnr = psnr(arr_gt, arr_output,data_range=1.)
        val_mssim = ssim(arr_gt, arr_output,data_range=1.)
        # metrices['psnr'].append(val_psnr); metrices['ssim'].append(val_mssim)
        metrices['psnr'].add(val_psnr); metrices['ssim'].add(val_mssim)
        # else: earlyexit_validate_loss(output, target, criterion, args)

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

        steps_completed = (validation_step+1)
        if epoch >= 0 and epoch % args.print_freq == 0 or is_last_epoch:
            _log_validation_progress()    

    # if not _is_earlyexit(args):
    if epoch >= 0 and epoch % args.print_freq == 0 or is_last_epoch:
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
    #    return losses_exits_stats[args.num_exits-1]
    return losses['objective_loss'].mean, metrices['psnr'].mean, metrices['ssim'].mean


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
    