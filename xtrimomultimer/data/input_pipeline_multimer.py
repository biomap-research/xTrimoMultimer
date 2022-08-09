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

from typing import Callable, List
from xtrimomultimer.data import (
    data_transforms,
    data_transforms_multimer,
)

BASE_NON_ENSEMBLED_TRANSFORMS = [
    data_transforms.cast_to_64bit_ints,
    data_transforms_multimer.make_msa_profile,
    data_transforms_multimer.create_target_feat,
    data_transforms.make_atom14_masks,
]

TEMPLATE_TRANSFORMS = [
    data_transforms.make_pseudo_beta("template_"),
]


def nonensembled_transform_fns(common_cfg, mode_cfg) -> List[Callable]:
    """Input pipeline data transformers that are not ensembled."""
    transforms = []
    transforms.extend(BASE_NON_ENSEMBLED_TRANSFORMS)

    if common_cfg.use_templates:
        transforms.extend(TEMPLATE_TRANSFORMS)

    return transforms


def ensembled_transform_fns(common_cfg, mode_cfg, ensemble_seed) -> List[Callable]:
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    pad_msa_clusters = mode_cfg.max_msa_clusters
    max_msa_clusters = pad_msa_clusters
    max_extra_msa = common_cfg.max_extra_msa

    msa_seed = None
    if not common_cfg.resample_msa_in_recycling:
        msa_seed = ensemble_seed

    transforms.append(
        data_transforms_multimer.sample_msa(
            max_msa_clusters,
            max_extra_msa,
            seed=msa_seed,
        )
    )

    if "masked_msa" in common_cfg:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms_multimer.make_masked_msa(
                common_cfg.masked_msa,
                mode_cfg.masked_msa_replace_fraction,
                seed=(msa_seed + 1) if msa_seed else None,
            )
        )

    transforms.append(data_transforms_multimer.nearest_neighbor_clusters())
    transforms.append(data_transforms_multimer.create_msa_feat)

    return transforms
