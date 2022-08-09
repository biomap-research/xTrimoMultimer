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

from xtrimomultimer.data import data_transforms

BASE_NON_ENSEMBLED_TRANSFORMS = [
    data_transforms.cast_to_64bit_ints,
    data_transforms.correct_msa_restypes,
    data_transforms.squeeze_features,
    data_transforms.randomly_replace_msa_with_unknown(0.0),
    data_transforms.make_seq_mask,
    data_transforms.make_msa_mask,
    data_transforms.make_hhblits_profile,
]

TEMPLATE_TRANSFORMS = [
    data_transforms.fix_templates_aatype,
    data_transforms.make_template_mask,
    data_transforms.make_pseudo_beta("template_"),
]

TEMPLATE_TORSION_TRANSFORMS = [data_transforms.atom37_to_torsion_angles("template_")]

ATOM_TRANSFORMS = [data_transforms.make_atom14_masks]

SUPERVISED_TRANSFORMS = [
    data_transforms.make_atom14_positions,
    data_transforms.atom37_to_frames,
    data_transforms.atom37_to_torsion_angles(""),
    data_transforms.make_pseudo_beta(""),
    data_transforms.get_backbone_frames,
    data_transforms.get_chi_angles,
]


def nonensembled_transform_fns(common_cfg, mode_cfg) -> List[Callable]:
    """Input pipeline data transformers that are not ensembled."""
    transforms = []
    transforms.extend(BASE_NON_ENSEMBLED_TRANSFORMS)
    if common_cfg.use_templates:
        transforms.extend(TEMPLATE_TRANSFORMS)
        if common_cfg.use_template_torsion_angles:
            transforms.extend(TEMPLATE_TORSION_TRANSFORMS)

    transforms.extend(ATOM_TRANSFORMS)

    if mode_cfg.supervised:
        transforms.extend(SUPERVISED_TRANSFORMS)

    return transforms


def ensembled_transform_fns(common_cfg, mode_cfg, ensemble_seed) -> List[Callable]:
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    if "max_distillation_msa_clusters" in mode_cfg:
        transforms.append(
            data_transforms.sample_msa_distillation(
                mode_cfg.max_distillation_msa_clusters
            )
        )

    if common_cfg.reduce_msa_clusters_by_max_templates:
        pad_msa_clusters = mode_cfg.max_msa_clusters - mode_cfg.max_templates
    else:
        pad_msa_clusters = mode_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters
    max_extra_msa = common_cfg.max_extra_msa

    msa_seed = None
    if not common_cfg.resample_msa_in_recycling:
        msa_seed = ensemble_seed

    transforms.append(
        data_transforms.sample_msa(max_msa_clusters, keep_extra=True, seed=msa_seed)
    )

    if "masked_msa" in common_cfg:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms.make_masked_msa(
                common_cfg.masked_msa, mode_cfg.masked_msa_replace_fraction
            )
        )

    if common_cfg.msa_cluster_features:
        transforms.extend(
            [
                data_transforms.nearest_neighbor_clusters(),
                data_transforms.summarize_clusters(),
            ]
        )

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(data_transforms.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(data_transforms.delete_extra_msa)

    transforms.append(data_transforms.make_msa_feat())

    crop_feats = dict(common_cfg.feat)

    if mode_cfg.fixed_size:
        transforms.extend(
            [
                data_transforms.select_feat(list(crop_feats)),
                data_transforms.random_crop_to_size(
                    mode_cfg.crop_size,
                    mode_cfg.max_templates,
                    crop_feats,
                    mode_cfg.subsample_templates,
                    seed=ensemble_seed + 1,
                ),
                data_transforms.make_fixed_size(
                    crop_feats,
                    pad_msa_clusters,
                    common_cfg.max_extra_msa,
                    mode_cfg.crop_size,
                    mode_cfg.max_templates,
                ),
            ]
        )
    else:
        transforms.append(data_transforms.crop_templates(mode_cfg.max_templates))

    return transforms
