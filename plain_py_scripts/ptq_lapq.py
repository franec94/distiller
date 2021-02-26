#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import torch
from copy import deepcopy
import logging
from collections import OrderedDict

import distiller
import distiller.apputils as apputils
# import distiller.apputils.image_classifier as classifier
import distiller.apputils.siren_image_regressor
import distiller.quantization.ptq_coordinate_search as lapq
import distiller.quantization


msglogger = logging.getLogger()

def save_arch_as_pytorch_model(quantizer, dummy_input, args):
    """Save quantized model as pytorch-compliant model within run output directory, of local file system.
    Arsg:
    -----
    `quantizer` - distiller.quantization.PostTrainLinearQuantizer object.\n
    `dummy_input` - model's dummy input.\n
    `args` - parser.Namespace object for getting output directory where model will be saved as pytorch-compliant arch.\n
    """
    # dummy_input = distiller.get_dummy_input(input_shape=model.input_shape)
    pyt_model = quantizer.convert_to_pytorch(dummy_input)
    pyt_model_path = os.path.join(args.output_dir, 'model_pyt.th')
    torch.save(pyt_model.state_dict(), pyt_model_path)
    pass


def image_regressor_ptq_lapq(model, criterion, loggers, args, scheduler=None, save_as_pytorch_model=False):
    """Post train quantization applied to a given distiller-based model"""
    args = deepcopy(args)

    effective_test_size_bak = args.effective_test_size
    args.effective_test_size = args.lapq_eval_size
    eval_data_loader = distiller.apputils.siren_image_regressor.load_data(args, load_train=False, load_val=False, load_test=True, fixed_subset=True)

    args.effective_test_size = effective_test_size_bak
    test_data_loader = distiller.apputils.siren_image_regressor.load_data(args, load_train=False, load_val=False, load_test=True)

    model = model.eval()
    device = next(model.parameters()).device

    if args.lapq_eval_memoize_dataloader:
        images_batches = []
        targets_batches = []
        for images, targets in eval_data_loader:
            images_batches.append(images.to(device))
            targets_batches.append(targets.to(device))
        memoized_data_loader = [(torch.cat(images_batches), torch.cat(targets_batches))]
    else:
        memoized_data_loader = None

    def eval_fn(model):
        if memoized_data_loader:
            loss = 0
            for images, targets in memoized_data_loader:
                outputs = model(images)
                loss += criterion(outputs, targets).item()
            loss = loss / len(memoized_data_loader)
        else:
            loss, psnr_score, ssim_score = distiller.apputils.siren_image_regressor.test(eval_data_loader, model, criterion, loggers, None, args)
        return loss

    def test_fn(model):
        loss, psnr_score, ssim_score = distiller.apputils.siren_image_regressor.test(test_data_loader, model, criterion, loggers, None, args)
        return OrderedDict([('loss_score', loss), ('psnr_score', psnr_score), ('ssim_score', ssim_score)])

    args.device = device
    if args.resumed_checkpoint_path:
        args.load_model_path = args.resumed_checkpoint_path
    if args.load_model_path:
        msglogger.info("Loading checkpoint from %s" % args.load_model_path)
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)

    quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(model, args)

    dummy_input = torch.rand(*model.input_shape, device=args.device)
    model, qp_dict = lapq.ptq_coordinate_search(quantizer, dummy_input, eval_fn, test_fn=test_fn,
                                                **lapq.cmdline_args_to_dict(args))

    results = test_fn(quantizer.model)
    msglogger.info("Arch: %s \tTest: \t loss_score = %.3f \t psnr_score = %.3f \t ssim_score = %.3f" %
                   (args.arch, results['loss_score'], results['psnr_score'], results['ssim_score']))
    distiller.yaml_ordered_save('%s.quant_params_dict.yaml' % args.arch, qp_dict)

    distiller.apputils.save_checkpoint(0, args.arch, model,
                                       extras={'loss_score': results['loss_score'], 'qp_dict': qp_dict}, name=args.name,
                                       dir=msglogger.logdir)
    if args.save_image_on_test:
        test_loader = distiller.apputils.siren_image_regressor.load_test_data(args)
        distiller.apputils.siren_image_regressor.save_predicted_data(test_loader, model, criterion, loggers,
            distiller.apputils.siren_image_regressor.create_activation_stats_collectors(model, *args.activation_stats),
                args, scheduler=scheduler)
        pass
    if save_as_pytorch_model:
        save_arch_as_pytorch_model(quantizer, dummy_input, args)
        pass
    pass


if __name__ == "__main__":
    parser = distiller.apputils.siren_image_regressor.init_classifier_compression_arg_parser(include_ptq_lapq_args=True)
    args = parser.parse_args()
    args.epochs = float('inf')  # hack for args parsing so there's no error in epochs
    cc = distiller.apputils.siren_image_regressor.SirenRegressorCompressor(args, script_dir=os.path.dirname(__file__))
    image_regressor_ptq_lapq(cc.model, cc.criterion, [cc.pylogger, cc.tflogger], cc.args)
