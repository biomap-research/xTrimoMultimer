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

import torch

JIT_OPTIONS_SET = False


def _set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    global JIT_OPTIONS_SET
    if JIT_OPTIONS_SET == False:
        # flags required to enable jit fusion kernels
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        # if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        #     # nvfuser
        #     torch._C._jit_set_profiling_executor(True)
        #     torch._C._jit_set_profiling_mode(True)
        #     torch._C._jit_override_can_fuse_on_cpu(False)
        #     torch._C._jit_override_can_fuse_on_gpu(False)
        #     torch._C._jit_set_texpr_fuser_enabled(False)
        #     torch._C._jit_set_nvfuser_enabled(True)
        #     torch._C._debug_set_autodiff_subgraph_inlining(False)
        # else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

        JIT_OPTIONS_SET = True
