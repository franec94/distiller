#!/usr/bin/env python3
 # -*- coding: utf-8 -*-

# Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

from src.libs import *

import traceback
import logging
from functools import partial
import distiller
from distiller.models import create_model
import distiller.apputils.siren_image_regressor
import distiller.apputils
import examples.siren_compression.custom_parser 
import os
import numpy as np
from ptq_lapq import image_regressor_ptq_lapq

from distiller.models import register_user_model
from distiller.models.siren import Siren
import distiller.apputils.siren_image_regressor as regressor


# Logger handle
msglogger = logging.getLogger()


def init_regressor_default_args(args, opt):
    """Init args Namespaceobject to be employed by regression class for representing input image by an inplicit representation.
    Args
    ----
    `args` - Namespace object to be correctly configurated.\n
    `opt` - Namespace object from script used to provide or supply some arguments collected from cmd line to Regression Class args.\n
    """
    args.seed = opt.seed[0]
    args.deterministic = True
    args.load_serialized = False
    args.deprecated_resume = None
    args.resumed_checkpoint_path = opt.resumed_checkpoint_path
    args.load_model_path = None
    args.reset_optimizer = opt.reset_optimizer
    args.lr = args.momentum = args.weight_decay = 0.0
    args.epochs = 0
    args.activation_stats = list()
    args.batch_size = 1
    args.workers = 1
    args.validation_split = 0.1
    args.effective_train_size = args.effective_valid_size = args.effective_test_size = 1.
    args.log_params_histograms = False
    args.print_freq = opt.epochs_til_ckpt
    args.masks_sparsity = False
    args.display_confusion = False
    args.num_best_scores = opt.num_best_scores
    args.name = ""
    args.kd_policy = None
    args.summary = opt.summary
    args.qe_stats_file = None
    args.qe_calibration = opt.qe_calibration
    args.verbose = True
    args.save_mid_ckpts = opt.save_mid_ckpts
    args.save_image_on_test = opt.save_image_on_test
    args.mid_target_sparsities = opt.mid_target_sparsities

    if opt.logging_root == '':
        args.output_dir = None
    else:
        args.output_dir = str(opt.logging_root)
    if opt.cuda:
        args.cpu = False
        args.gpus = "0"
    else:
        args.cpu = True
        args.gpus = ""
    if opt.compress == None or opt.compress == '':
        args.compress = None
    else:
        args.compress = opt.compress
    if opt.sensitivity != None:
        args.sensitivity = opt.sensitivity
        args.sensitivity_range = opt.sensitivity_range
    if opt.load_model_path:
        args.load_model_path = opt.load_model_path
    args.evaluate = False
    if opt.evaluate and opt.train == False:
        args.evaluate = opt.evaluate

    args = _config_early_stopping_sparsity(args, opt)
    args = _config_ptq(args, opt)
    return args


def _config_early_stopping_sparsity(args, opt):
    """Update args keeping run config for holding options for employing if necessary early stopping algorithm for sparsity monitoring.
    Args
    ----
    `args` - Namespace object to be correctly configurated.\n
    `opt` - Namespace object from script used to provide or supply some arguments collected from cmd line to Regression Class args.\n
    """
    args.target_sparsity = opt.target_sparsity
    args.toll_sparsity = opt.toll_sparsity
    args.patience_sparsity = opt.patience_sparsity
    args.trail_epochs = opt.trail_epochs
    return args


def _config_ptq(args, opt):
    """Update args keeping run config for holding options for post train quant.
    Args
    ----
    `args` - Namespace object to be correctly configurated.\n
    `opt` - Namespace object from script used to provide or supply some arguments collected from cmd line to Regression Class args.\n
    """
    args.qe_lapq = opt.qe_lapq
    args.quantize_eval = opt.quantize_eval
    args.qe_config_file = opt.qe_config_file
    return args


def config_learner_args(args, arch, dataset, dataset_path, pretrained, adam_args, batch, epochs):
    args.arch = f"{arch}"
    args.dataset = f"{dataset}"
    args.data = ""
    args.pretrained = False
    args.lr = adam_args[0]
    args.momentum = adam_args[1]
    args.weight_decay = adam_args[2]
    args.batch_size = 1
    args.epochs = epochs
    return args


def main(opt):
    
    # Register custom model
    def sire_model(n_hf = opt.n_hf[0], n_hl = opt.n_hl[0]):
        img_siren = Siren(in_features=2, out_features=1, hidden_features=n_hf, 
                  hidden_layers=n_hl, outermost_linear=True)
        return img_siren

    distiller.models.register_user_model(
        arch="SirenCompressingModel",
        dataset="cameramen",
        model=sire_model)
    # Parse arguments
    args , _ = examples.siren_compression.custom_parser.add_cmdline_args(distiller.apputils.siren_image_regressor.init_regressor_compression_arg_parser(True)).parse_known_args() # .parse_args()
    args = init_regressor_default_args(args, opt)
    args = config_learner_args(
        args,
        "SirenCompressingModel", "cameramen", "",
        False,
        (opt.lr[0], opt.momentum[0], opt.lambda_L_2[0]),
        1,
        opt.num_epochs[0])
    
    # app = SirenRegressorCompressorSampleApp(args, script_dir=os.path.dirname(__file__))
    app = SirenRegressorCompressorSampleApp(args, script_dir=opt.logging_root)
    if app.handle_subapps():
        return
    if opt.train:
        init_knowledge_distillation(app.args, app.model, app.compression_scheduler)
        app.run_training_loop()

        # Finally run results on the test set
        if opt.evaluate:
            return app.test()
        pass
    return

    
def handle_subapps(model, criterion, optimizer, compression_scheduler, pylogger, args):
    def load_test_data(args):
        test_loader = distiller.apputils.siren_image_regressor.load_data(args, load_train=False, load_val=False, load_test=True)
        return test_loader

    do_exit = False
    if args.greedy:
        greedy(model, criterion, optimizer, pylogger, args)
        do_exit = True
    elif args.summary:
        # This sample application can be invoked to produce various summary reports
        for summary in args.summary:
            distiller.model_summary(model, summary, args.dataset)
        do_exit = True
    elif args.export_onnx is not None:
        distiller.export_img_classifier_to_onnx(model,
                                                os.path.join(msglogger.logdir, args.export_onnx),
                                                args.dataset, add_softmax=True, verbose=False)
        do_exit = True
    elif args.qe_calibration and not (args.evaluate and args.quantize_eval):
        distiller.apputils.siren_image_regressor.acts_quant_stats_collection(model, criterion, pylogger, args, save_to_file=True)
        do_exit = True
    elif args.activation_histograms:
        distiller.apputils.siren_image_regressor.acts_histogram_collection(model, criterion, pylogger, args)
        do_exit = True
    elif args.sensitivity is not None:
        test_loader = load_test_data(args)
        sensitivities = np.arange(*args.sensitivity_range)
        sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)
        do_exit = True
    elif args.evaluate:
        if args.quantize_eval and args.qe_lapq:
            image_regressor_ptq_lapq(
                model, criterion, pylogger, args,
                scheduler=compression_scheduler, save_as_pytorch_model=True)
            pass
        else:
            test_loader = load_test_data(args)
            distiller.apputils.siren_image_regressor.evaluate_model(test_loader, model, criterion, pylogger,
                distiller.apputils.siren_image_regressor.create_activation_stats_collectors(model, *args.activation_stats),
                args, scheduler=compression_scheduler)
            if args.save_image_on_test:
                test_loader = load_test_data(args)
                distiller.apputils.siren_image_regressor.save_predicted_data(test_loader, model, criterion, pylogger,
                    distiller.apputils.siren_image_regressor.create_activation_stats_collectors(model, *args.activation_stats),
                    args, scheduler=compression_scheduler)
        do_exit = True
    elif args.thinnify:
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.contract_model(model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
        distiller.apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        msglogger.info("Note: if your model collapsed to random inference, you may want to fine-tune")
        do_exit = True
    return do_exit


def init_knowledge_distillation(args, model, compression_scheduler):
    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
        if args.kd_resume:
            teacher = distiller.apputils.load_lean_checkpoint(teacher, args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)
        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)


def early_exit_init(args):
    if not args.earlyexit_thresholds:
        return
    args.num_exits = len(args.earlyexit_thresholds) + 1
    args.loss_exits = [0] * args.num_exits
    args.losses_exits = []
    args.exiterrors = []
    msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)


class SirenRegressorCompressorSampleApp(distiller.apputils.siren_image_regressor.SirenRegressorCompressor):
    def __init__(self, args, script_dir):
        super().__init__(args, script_dir)
        early_exit_init(self.args)
        # Save the randomly-initialized model before training (useful for lottery-ticket method)
        if args.save_untrained_model:
            ckpt_name = '_'.join((self.args.name or "", "untrained"))
            distiller.apputils.save_checkpoint(0, self.args.arch, self.model,
                                     name=ckpt_name, dir=msglogger.logdir)


    def handle_subapps(self):
        return handle_subapps(self.model, self.criterion, self.optimizer,
                              self.compression_scheduler, self.pylogger, self.args)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(distiller.apputils.siren_image_regressor.test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=distiller.apputils.siren_image_regressor.create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity,
                                                         kind_task = 'regression')
    distiller.sensitivities_to_png(sensitivity, os.path.join(msglogger.logdir, 'sensitivity.png'), kind_task = 'regression')
    distiller.sensitivities_to_csv(sensitivity, os.path.join(msglogger.logdir, 'sensitivity.csv'), kind_task = 'regression')


def greedy(model, criterion, optimizer, loggers, args):
    train_loader, val_loader, test_loader = distiller.apputils.siren_image_regressor.load_data(args)

    test_fn = partial(distiller.apputils.siren_image_regressor.test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(distiller.apputils.siren_image_regressor.train, train_loader=train_loader, criterion=criterion, args=args)
    assert args.greedy_target_density is not None
    distiller.pruning.greedy_filter_pruning.greedy_pruner(model, args,
                                                          args.greedy_target_density,
                                                          args.greedy_pruning_step,
                                                          test_fn, train_fn)


if __name__ == '__main__':
    try:
        opt, parser = get_cmd_line_opts()
        main(opt)
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
