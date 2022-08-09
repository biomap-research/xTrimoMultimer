# Copyright 2021 AlQuraishi Laboratory
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
import pytest
import numpy as np

from xtrimomultimer.config import model_config
from xtrimomultimer.model.model import AlphaFold
from xtrimomultimer.utils.import_weights import import_jax_weights_

JAX_MULTIMER_WEIGHTS_FILE = "/database/multimer/params/params_model_1_ptm.npz"


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in JAX_MULTIMER_WEIGHTS_FILE),
    reason="jax_multimer_weight_file doesn't exist",
)
def test_import_jax_weights_():
    npz_path = "xtrimomultimer/resources/params/params_model_1_ptm.npz"
    if not os.path.exists(npz_path):
        npz_path = JAX_MULTIMER_WEIGHTS_FILE

    c = model_config("model_1_ptm")
    c.globals.blocks_per_ckpt = None
    model = AlphaFold(c)

    import_jax_weights_(
        model,
        npz_path,
    )

    data = np.load(npz_path)
    prefix = "alphafold/alphafold_iteration/"

    test_pairs = [
        # Normal linear weight
        (
            torch.as_tensor(
                data[prefix + "structure_module/initial_projection//weights"]
            ).transpose(-1, -2),
            model.structure_module.linear_in.weight,
        ),
        # Normal layer norm param
        (
            torch.as_tensor(
                data[prefix + "evoformer/prev_pair_norm//offset"],
            ),
            model.recycling_embedder.layer_norm_z.bias,
        ),
        # From a stack
        (
            torch.as_tensor(
                data[
                    prefix
                    + (
                        "evoformer/evoformer_iteration/outer_product_mean/"
                        "left_projection//weights"
                    )
                ][1].transpose(-1, -2)
            ),
            model.evoformer.blocks[1].core.outer_product_mean.linear_1.weight,
        ),
    ]

    for w_alpha, w_repro in test_pairs:
        assert torch.all(w_alpha == w_repro)
