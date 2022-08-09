/**
    Copyright 2022 BioMap (Beijing) Intelligence Technology Limited
    Copyright 2022 HPC-AI Technology Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
**/

#include <torch/extension.h>
#include "softmax_cuda.h"
#include "layer_norm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_forward", &softmax, "Softmax forward (CUDA)");
    m.def("softmax_backward", &softmax_gradient, "Softmax backward (CUDA)");

    m.def("fused_scale_mask_softmax_forward", &fused_scale_mask_softmax_forward,
          "Softmax forward (CUDA)");
    m.def("fused_scale_mask_softmax_backward", &fused_scale_mask_softmax_backward,
          "Softmax forward (CUDA)");

    m.def("fused_scale_mask_bias_softmax_forward", &fused_scale_mask_bias_softmax_forward,
          "Softmax forward (CUDA)");
    m.def("fused_scale_mask_bias_softmax_backward", &fused_scale_mask_bias_softmax_backward,
          "Softmax forward (CUDA)");

    m.def("layer_normal_forward_affine", &layer_norm_affine, "LayerNorm forward (CUDA)");
    m.def("layer_normal_backward_affine", &layer_norm_gradient_affine, "LayerNorm backward (CUDA)");
}
