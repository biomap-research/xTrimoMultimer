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

# -*-coding:utf-8-*-
import logging
import logging.config
import time

from pytorch_lightning.utilities.distributed import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance()


@singleton
class Logger:
    def __init__(self):
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d] %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("root")

    def set_logger(self, env="terminal"):
        """
        default evn is terminal
        when we connect our agent with web service, we need to automatic output all logger into one file, pls set evn='web'
        :param env:
        :return:
        """
        if env == "terminal":
            logging.basicConfig(
                format="%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d] %(message)s",
                level=logging.INFO,
            )
            self.logger = logging.getLogger("root")
        else:
            self.logger = logging.getLogger("root")
            self.logger.setLevel(logging.INFO)

            BASIC_FORMAT = "%(asctime)s %(levelname)s %(process)d [%(filename)s:%(lineno)d] %(message)s"
            formatter = logging.Formatter(BASIC_FORMAT)

            # stdout
            chlr = logging.StreamHandler()
            chlr.setLevel(logging.INFO)
            chlr.setFormatter(formatter)
            # log file
            file_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".log"
            fhlr = logging.FileHandler(file_name)
            fhlr.setLevel(logging.INFO)
            fhlr.setFormatter(formatter)

            self.logger.addHandler(chlr)
            self.logger.addHandler(fhlr)
