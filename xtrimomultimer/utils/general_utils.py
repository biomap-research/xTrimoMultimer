# Copyright 2022 BioMap (Beijing) Intelligence Technology Limited
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

"""Common utilities for data pipeline tools."""
import contextlib
import datetime
import shutil
import tempfile
import time
from typing import Optional
import torch

from xtrimomultimer.utils.logger import Logger

logger = Logger.logger


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logger.info("Started %s", msg)
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    logger.info("Finished %s in %.3f seconds", msg, toc - tic)


def to_date(s: str) -> datetime.datetime:
    return datetime.datetime(year=int(s[:4]), month=int(s[5:7]), day=int(s[8:10]))


def warmup_gpu_for_pytorch(device) -> torch.Tensor:
    torch.tensor([0]).to(device)
