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
import distiller
from distiller.models import register_user_model
from distiller.models.siren import Siren
import distiller.apputils.siren_image_regressor as regressor


def init_jupyter_default_args(args, opt):

    if opt.logging_root == '':
        args.output_dir = None
    else:
        args.output_dir = str(opt.logging_root)
    args.evaluate = False
    args.seed = opt.seed[0]
    args.deterministic = True
    args.cpu = False
    args.gpus = "0"
    args.load_serialized = False
    args.deprecated_resume = None
    args.resumed_checkpoint_path = None
    args.load_model_path = None
    args.reset_optimizer = False
    args.lr = args.momentum = args.weight_decay = 0.
    if opt.compress == None or opt.compress == '':
        args.compress = None
    else:
        args.compress = opt.compress
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
    args.num_best_scores = 1
    args.name = ""
    args.kd_policy = None
    # args.summary = "sparsity"
    args.qe_stats_file = None
    args.verbose = True
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

    def sire_model(n_hf = opt.n_hf[0], n_hl = opt.n_hl[0]):
        img_siren = Siren(in_features=2, out_features=1, hidden_features=n_hf, 
                  hidden_layers=n_hl, outermost_linear=True)
        return img_siren

    distiller.models.register_user_model(
        arch="SirenCompressingModel",
        dataset="cameramen",
        model=sire_model)

    args = regressor.init_regressor_compression_arg_parser()
    args, _ = args.parse_known_args()

    args = init_jupyter_default_args(args, opt)
    args = config_learner_args(
        args,
        "SirenCompressingModel", "cameramen", "",
        False,
        (opt.lr[0], opt.momentum[0], opt.lambda_L_2[0]),
        1,
        opt.num_epochs[0])
    # app = regressor.SirenRegressorCompressor(args, script_dir=os.path.dirname("."))
    app = regressor.SirenRegressorCompressor(args, script_dir=opt.logging_root)
    _ = app.run_training_loop()
    pass


if __name__ == "__main__":

    opt, parser = get_cmd_line_opts()

    # Set seeds for experiment re-running.
    if hasattr(opt, 'seed'): seed = opt.seed[0]
    else: seed = 0
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(opt)
    pass