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

from typing import Callable, List
import torch

from xtrimomultimer.data.data_transforms import curry1


@curry1
def compose(x, fs: List[Callable]):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun: Callable, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict


def process_tensors_from_config(tensors, common_cfg, mode_cfg, is_multimer: bool):
    """Based on the config, apply filters and transformations to the data."""

    ensemble_seed = torch.Generator().seed()

    if is_multimer:
        from xtrimomultimer.data.input_pipeline_multimer import (
            ensembled_transform_fns,
            nonensembled_transform_fns,
        )
    else:
        from xtrimomultimer.data.input_pipeline_monomer import (
            ensembled_transform_fns,
            nonensembled_transform_fns,
        )

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(
            common_cfg,
            mode_cfg,
            ensemble_seed,
        )
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    # TODO: unused no_templates identifier
    no_templates = True
    if "template_aatype" in tensors:
        no_templates = tensors["template_aatype"].shape[0] == 0

    nonensembled = nonensembled_transform_fns(
        common_cfg,
        mode_cfg,
    )

    tensors = compose(nonensembled)(tensors)

    if "no_recycling_iters" in tensors:
        num_recycling = int(tensors["no_recycling_iters"])
    else:
        num_recycling = common_cfg.max_recycling_iters

    tensors = map_fn(
        lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
    )

    return tensors
