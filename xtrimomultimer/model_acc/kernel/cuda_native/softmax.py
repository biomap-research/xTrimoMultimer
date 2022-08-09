# Copyright 2022 BioMap (Beijing) Intelligence Technology Limited
# Copyright 2022 HPC-AI Technology Inc.
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
import importlib
from functools import reduce
from operator import mul

import torch

fastfold_cuda_ops = importlib.import_module("fastfold_cuda_ops")


class SoftmaxAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input_ = input.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = fastfold_cuda_ops.softmax_forward(input_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output = ctx.saved_tensors[0]

        grad_input = None
        grad_input = fastfold_cuda_ops.softmax_backward(
            grad_output.contiguous(), output, ctx.rows, ctx.cols
        )

        return grad_input


class FusedScaleMaskSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, scale):
        input_ = input.contiguous()
        mask_ = mask.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = fastfold_cuda_ops.fused_scale_mask_softmax_forward(
            input_, mask_, ctx.rows, ctx.cols, scale
        )
        ctx.save_for_backward(output, mask_)
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output, mask_ = ctx.saved_tensors

        grad_input = None
        grad_input = fastfold_cuda_ops.fused_scale_mask_softmax_backward(
            grad_output.contiguous(), output, mask_, ctx.rows, ctx.cols, ctx.scale
        )

        return grad_input.contiguous(), None, None


class FusedScaleMaskBiasSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, bias, scale):
        input_ = input.contiguous()
        mask_ = mask.contiguous()
        bias_ = bias.contiguous()
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = fastfold_cuda_ops.fused_scale_mask_bias_softmax_forward(
            input_, mask_, bias_, ctx.rows, ctx.cols, scale
        )
        ctx.save_for_backward(output, mask_, bias_)
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output):

        output, mask_, bias_ = ctx.saved_tensors

        grad_input = None
        grad_input = fastfold_cuda_ops.fused_scale_mask_bias_softmax_backward(
            grad_output.contiguous(),
            output,
            mask_,
            bias_,
            ctx.rows,
            ctx.cols,
            ctx.scale,
        )

        grad_input = grad_input.contiguous()

        grad_bias = torch.sum(grad_input, dim=1, keepdim=True)

        return grad_input.contiguous(), grad_bias, None, None


softmax = SoftmaxAffineFunction.apply
scale_mask_softmax = FusedScaleMaskSoftmaxFunction.apply
scale_mask_bias_softmax = FusedScaleMaskBiasSoftmaxFunction.apply
