# Copyright 2022 BioMap (Beijing) Intelligence Technology Limited
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

import random
import numpy as np
import torch


def set_device_seed(args):
    torch.set_num_threads(1)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")
    # torch.multiprocessing.set_start_method('spawn')  # enable multiprocess for torch on cuda
    if args.cuda and torch.cuda.is_available():
        """reproduce on cuda, it will make the program slower."""
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return device
