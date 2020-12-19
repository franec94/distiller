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


def init_regressor_compression_arg_parser(include_ptq_lapq_args=False):
    '''Common classifier-compression application command-line arguments.
    '''
    SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params']

    parser_regressor = argparse.ArgumentParser(description='Distiller image classification model compression')
    parser_regressor.add_argument('data', metavar='DATASET_DIR', help='path to dataset')
    parser_regressor.add_argument('--arch', '-a', metavar='ARCH', default='siren', type=lambda s: s.lower(),
                        choices=distiller.models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(distiller.models.ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
    parser_regressor.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser_regressor.add_argument('--epochs', type=int, metavar='N', default=90,
                        help='number of total epochs to run (default: 90')
    parser_regressor.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')

    parser_regressor.add_argument('-n_hf', '--number-hidden-features', default=8, type=int,
                        metavar='N', help='number hidden features (default: 8)')
    parser_regressor.add_argument('-n_hl', '--number-hidden-layerss', default=8, type=int,
                        metavar='N', help='number hidden layers (default: 8)')

    optimizer_args = parser_regressor.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
    optimizer_args.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

    parser_regressor.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser_regressor.add_argument('--verbose', '-v', action='store_true', default=False, help='Emit debug log messages')

    load_checkpoint_group = parser_regressor.add_argument_group('Resuming arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    load_checkpoint_group_exc.add_argument('--resume', dest='deprecated_resume', default='', type=str,
                        metavar='PATH', help=argparse.SUPPRESS)
    load_checkpoint_group_exc.add_argument('--resume-from', dest='resumed_checkpoint_path', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint. Use to resume paused training session.')
    load_checkpoint_group_exc.add_argument('--exp-load-weights-from', dest='load_model_path',
                        default='', type=str, metavar='PATH',
                        help='path to checkpoint to load weights from (excluding other fields) (experimental)')
    load_checkpoint_group.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    load_checkpoint_group.add_argument('--reset-optimizer', action='store_true',
                        help='Flag to override optimizer if resumed from checkpoint. This will reset epochs count.')

    parser_regressor.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser_regressor.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                        help='collect activation statistics on phases: train, valid, and/or test'
                        ' (WARNING: this slows down training)')
    parser_regressor.add_argument('--activation-histograms', '--act-hist',
                        type=float_range(exc_min=True),
                        metavar='PORTION_OF_TEST_SET',
                        help='Run the model in evaluation mode on the specified portion of the test dataset and '
                             'generate activation histograms. NOTE: This slows down evaluation significantly')
    parser_regressor.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                        help='print masks sparsity table at end of each epoch')
    parser_regressor.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file '
                             '(WARNING: this can use significant disk space)')
    parser_regressor.add_argument('--summary', type=lambda s: s.lower(), choices=SUMMARY_CHOICES, action='append',
                        help='print a summary of the model, and exit - options: | '.join(SUMMARY_CHOICES))
    parser_regressor.add_argument('--export-onnx', action='store', nargs='?', type=str, const='model.onnx', default=None,
                        help='export model to ONNX format')
    parser_regressor.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')
    parser_regressor.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                        type=lambda s: s.lower(), help='test the sensitivity of layers to pruning')
    parser_regressor.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3, default=[0.0, 0.95, 0.05],
                        help='an optional parameter for sensitivity testing '
                             'providing the range of sparsities to test.\n'
                             'This is equivalent to creating sensitivities = np.arange(start, stop, step)')
    parser_regressor.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')
    parser_regressor.add_argument('--seed', type=int, default=None,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
    parser_regressor.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used '
                             '(default is to use all available devices)')
    parser_regressor.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                        'Flag not set => uses GPUs according to the --gpus flag value.'
                        'Flag set => overrides the --gpus flag')
    parser_regressor.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser_regressor.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
    parser_regressor.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                        type=float_range(exc_max=True), default=0.1,
                        help='Portion of training dataset to set aside for validation')
    parser_regressor.add_argument('--effective-train-size', '--etrs', type=float_range(exc_min=True), default=1.,
                        help='Portion of training dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser_regressor.add_argument('--effective-valid-size', '--evs', type=float_range(exc_min=True), default=1.,
                        help='Portion of validation dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument is applied '
                             'AFTER the train-validation split according to that argument')
    parser_regressor.add_argument('--effective-test-size', '--etes', type=float_range(exc_min=True), default=1.,
                        help='Portion of test dataset to be used in each epoch')
    parser_regressor.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                        help='Display the confusion matrix')
    parser_regressor.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                        help='number of best scores to track and report (default: 1)')
    parser_regressor.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                        help='Load a model without DataParallel wrapping it')
    parser_regressor.add_argument('--thinnify', dest='thinnify', action='store_true', default=False,
                        help='physically remove zero-filters and create a smaller model')

    # Added arguments with respect to original minima arguments for running trials
    # with this class.
    parser_regressor.add_argument('--save_mid_ckpts', nargs='+', type=int, default=[], dest = "save_mid_ckpts",
               help='Fixed desired checkpoints to be saved, at a given epoch, a part from default saving checkpoint system. Default empty list, meaning no intermediate checkpoints')
    parser_regressor.add_argument('--save-image-on-test', dest='save_image_on_test', action='store_true',
                        help='set it to save predicted image as png.')
    
    parser_regressor.add_argument('--target_sparsity', dest='target_sparsity', type=float, default=None,
                        help='Target sparsity, if None no earlystopping on sparsity is exploited.')
    parser_regressor.add_argument('--toll_sparsity', dest='toll_sparsity', type=float, default=2.0,
                        help='Target toll sparsity.')
    parser_regressor.add_argument('--patience_sparsity', dest='patience_sparsity', type=float, default=5,
                        help='Target patience sparsity.')
    parser_regressor.add_argument('--trail_epochs', dest='trail_epochs', type=float, default=5,
                        help='Target trail epochs sparsity.')
    parser_regressor.add_argument('--mid_target_sparsities', nargs='+', dest='mid_target_sparsities', type=float, default=[],
                        help='Target sparsities to save a part.')
    parser_regressor.add_argument("--wandb_logging", required=False, action="store_true", default=False, dest='wandb_logging',
        help="Flag for enabling model's performance, metrics via wandb API."
    )

    distiller.quantization.add_post_train_quant_args(parser_regressor, add_lapq_args=include_ptq_lapq_args)
    return parser_regressor


def _init_logger(args, script_dir, msglogger):
    """Init logger for keep track of operations and computatins done during a run.
    Args:
    ----
    `args` - Namespace compliant python object with details needed to determine how logger should be properly setup.\n
    `script_dir` - either None or str python object which refers to local file system path of directory where data and logging infos should be stored in.\n
    `msglogger` - logger python object or instance for logging operations we aim at tracking.\n
    Return:
    ------
    `logdir` -  local file system path to directory where data related to logged information will be collected.\n
    """
    # global msglogger
    if script_dir is None or not hasattr(args, "output_dir") or args.output_dir is None:
        msglogger.logdir = None
        return None
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = distiller.apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'),
                                         args.name, args.output_dir, args.verbose)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    distiller.apputils.log_execution_env_state(
        filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
        msglogger.logdir)
    msglogger.debug("Distiller: %s", distiller.__version__)
    return msglogger.logdir


def _init_wandb(args, msglogger = None):
    """Config wandb access for registering training progresses storing them to cloud by means of wandb API.
    Args:
    -----
    `args` - Namespace compliant python object with details needed to determine how data will be stored via wandb API to cloud storage.\n
    `msglogger` - logger python object or instance for logging operations we aim at tracking.\n
    """
    # global msglogger
    if args.wandb_logging:
        wandb.init(project='siren-run')


def _config_determinism(args, msglogger = None):
    """Config determines if requested for current run.
    Args:
    -----
    `args` - Namespace compliant python object with details needed to determine whether determinism has been required for current run.\n
    `msglogger` - logger python object or instance for logging operations we aim at tracking.\n
    """
    if args.evaluate:
        args.deterministic = True
    
    # Configure some seed (in case we want to reproduce this experiment session)
    if args.seed is None:
        if args.deterministic:
            args.seed = 0
        else:
            args.seed = np.random.randint(1, 100000)

    if args.deterministic:
        distiller.set_deterministic(args.seed) # For experiment reproducability
    else:
        distiller.set_seed(args.seed)
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True
    if msglogger:
        msglogger.info("=> Random seed: %d", args.seed)


def _config_compute_device(args, msglogger = None):
    """Config compute device: either CPU or GPU from CUDA engine.
    Args:
    -----
    `args` - Namespace compliant python object with details needed to determine to which engine and backend training/val/test (generally speaking any kind of computation) will be performed.\n
    `msglogger` - logger python object or instance for logging operations we aim at tracking.\n
    """
    # global msglogger
    if args.cpu or not torch.cuda.is_available():
        if msglogger:
            msglogger.info(f"=> Selected device: {args.device}, since args.cpu={args.cpu} or torch.cuda.is_available()={torch.cuda.is_available()}")
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                     .format(dev_id, available_gpus))
            # Set default device in case the first one on the list != 0
            if msglogger:
                msglogger.info(f"=> Selected device: {args.device}, since args.cpu={args.cpu} or torch.cuda.is_available()={torch.cuda.is_available()}")
                msglogger.info(f"=> Selected device: cuda, selected gpu_id={args.gpus[0]}")
            torch.cuda.set_device(args.gpus[0])
    return 


def _init_learner(args, msglogger = None):
    """Config learner instance to either be runned on CPU or GPU from CUDA engine.
    Args:
    -----
    `args` - Namespace compliant python object with details needed to determine how learner model object will be setup also from information provided from command line by user.\n
    `msglogger` - logger python object or instance for logging operations we aim at tracking.\n
    """
    model = None
    compression_scheduler = None
    optimizer = None
    start_epoch = 0

    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    
    target_device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    target_device_id = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        if msglogger:
            msglogger.warning(f'=> Model has been loaded to device={target_device}, with id number={target_device_id}')
    else:
        if msglogger:
            msglogger.warning(f'=> Model has been loaded to device={target_device}')
    
    
    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    if args.deprecated_resume:
        msglogger.warning('The "--resume" flag is deprecated. Please use "--resume-from=YOUR_PATH" instead.')
        if not args.reset_optimizer:
            msglogger.warning('If you wish to also reset the optimizer, call with: --reset-optimizer')
            args.reset_optimizer = True
        args.resumed_checkpoint_path = args.deprecated_resume

    if args.resumed_checkpoint_path:
        model, compression_scheduler, optimizer, start_epoch = distiller.apputils.load_checkpoint(
            model, args.resumed_checkpoint_path, model_device=args.device)
        if args.lr != -1.0:
            dest_state_dict = optimizer.state_dict()
            dest_state_dict['param_groups'][0]['lr'] = args.lr
            optimizer.load_state_dict(dest_state_dict)
            if msglogger:
                msglogger.debug('=> Optimizer LR updated: %f', optimizer.state_dict()['param_groups'][0]['lr'] )
    elif args.load_model_path:
        model = distiller.apputils.load_lean_checkpoint(model, args.load_model_path, model_device=args.device)
    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            if msglogger:
                msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')

    if optimizer is None and not args.evaluate:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
        if msglogger:
            msglogger.debug('Optimizer Type(created witout eval flag): %s', type(optimizer))
            msglogger.debug('Optimizer Args(created witout eval flag): %s', optimizer.defaults)
    elif optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
        if msglogger:
            msglogger.debug('Optimizer Type(created regardless of eval flag has been set): %s', type(optimizer))
            msglogger.debug('Optimizer Args(created regardless of eval flag has been set): %s', optimizer.defaults)
        

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler,
            (start_epoch-1) if args.resumed_checkpoint_path else None)
        # pprint(compression_scheduler.policies)
        # pprint(compression_scheduler.sched_metadata)
        # pprint(compression_scheduler.sched_metadata.keys()[0])
        # sys.exit(0)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        if args.lr != -1.0:
            optimizer.lr = args.lr
            if msglogger:
                msglogger.debug('=> Optimizer LR updated: %.2f', optimizer.lr )
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    return model, compression_scheduler, optimizer, start_epoch, args.epochs
