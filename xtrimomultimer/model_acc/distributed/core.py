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
import os

import torch
import colossalai


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def set_missing_distributed_environ(key, value):
    if key not in os.environ:
        os.environ[str(key)] = str(value)


def init_dap(tensor_model_parallel_size_=None):
    # colossalai.logging.disable_existing_loggers()

    if tensor_model_parallel_size_ == None:
        if "WORLD_SIZE" in os.environ:
            tensor_model_parallel_size_ = int(os.environ["WORLD_SIZE"])
        else:
            tensor_model_parallel_size_ = 1

    if torch.distributed.is_initialized():
        _logger = colossalai.logging.get_dist_logger()
        _logger.error(
            "use fastfold.distributed.init_dap instead of torch.distributed.init_process_group!"
        )
        exit(-1)

    # set distributed environ for single device launch
    set_missing_distributed_environ("WORLD_SIZE", 1)
    set_missing_distributed_environ("RANK", 0)
    set_missing_distributed_environ("LOCAL_RANK", 0)
    set_missing_distributed_environ("MASTER_ADDR", "localhost")
    set_missing_distributed_environ("MASTER_PORT", 18417)

    colossalai.launch_from_torch(
        config={"parallel": dict(tensor=dict(size=tensor_model_parallel_size_))}
    )
