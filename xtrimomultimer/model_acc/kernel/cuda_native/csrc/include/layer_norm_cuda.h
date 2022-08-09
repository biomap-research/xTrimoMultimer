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

std::vector<at::Tensor> layer_norm_affine(at::Tensor input, at::IntArrayRef normalized_shape,
                                          at::Tensor gamma, at::Tensor beta, double epsilon);

std::vector<at::Tensor> layer_norm_gradient_affine(at::Tensor dout, at::Tensor mean,
                                                   at::Tensor invvar, at::Tensor input,
                                                   at::IntArrayRef normalized_shape,
                                                   at::Tensor gamma, at::Tensor beta,
                                                   double epsilon);
