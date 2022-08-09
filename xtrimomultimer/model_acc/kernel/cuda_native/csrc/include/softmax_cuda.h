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

at::Tensor softmax(at::Tensor input, long long rows, long long cols);
at::Tensor softmax_gradient(at::Tensor d_output, at::Tensor output, long long rows, long long cols);

at::Tensor fused_scale_mask_softmax_forward(at::Tensor input, at::Tensor mask, long long rows, long long cols,
                                            float scale);
at::Tensor fused_scale_mask_softmax_backward(at::Tensor d_output, at::Tensor input, at::Tensor mask,
                                             long long rows, long long cols, float scale);

at::Tensor fused_scale_mask_bias_softmax_forward(at::Tensor input, at::Tensor mask, at::Tensor bias,
                                                 long long rows, long long cols, float scale);
at::Tensor fused_scale_mask_bias_softmax_backward(at::Tensor d_output, at::Tensor input,
                                                  at::Tensor mask, at::Tensor bias, long long rows,
                                                  long long cols, float scale);
