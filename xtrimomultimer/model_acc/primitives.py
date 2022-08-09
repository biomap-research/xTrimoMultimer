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
import math

import numpy as np
import torch.nn as nn


def glorot_uniform_af(x, gain=1.0):
    """
    initialize tensors the same as xavier_initializer in PyTorch, but the dimensions are different:
    In PyTorch:
    [feature_out, feature_in, n_head ...]
    In Jax:
    [... n_head, feature_in, feature_out]
    However, there is a feature in original Alphafold2 code that they use the Jax version initializer to initialize tensors like:
    [feature_in, n_head, feature_out]

    In this function, we keep this feature to initialize [feature_in, n_head, ..., feature_out] tensors
    """
    fan_in, fan_out = x.shape[-2:]
    if len(x.shape) > 2:
        receptive_field_size = np.prod(x.shape[:-2])
        fan_in *= receptive_field_size
        fan_out *= receptive_field_size
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    dev = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    nn.init.uniform_(x, -dev, dev)

    return x
