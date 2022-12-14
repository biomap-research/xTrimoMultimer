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

from typing import Any, Dict
import torch
import torch.nn as nn

from xtrimomultimer.data import data_transforms_multimer
from xtrimomultimer.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    atom14_to_atom37,
)
from xtrimomultimer.model.embedders import (
    InputEmbedder,
    InputEmbedderMultimer,
    RecyclingEmbedder,
    TemplateEmbedder,
    TemplateEmbedderMultimer,
    ExtraMSAEmbedder,
)
from xtrimomultimer.model.evoformer import EvoformerStack, ExtraMSAStack
from xtrimomultimer.model.heads import AuxiliaryHeads
import xtrimomultimer.np.residue_constants as residue_constants
from xtrimomultimer.model.structure_module import StructureModule

from xtrimomultimer.utils.tensor_utils import (
    tensor_tree_map,
)

from xtrimomultimer.utils.logger import Logger

logger = Logger.logger


class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        config = config.model
        template_config = config.template
        extra_msa_config = config.extra_msa

        # Main trunk + structure module
        if self.globals.is_multimer:
            self.input_embedder = InputEmbedderMultimer(
                **config["input_embedder"],
            )
        else:
            self.input_embedder = InputEmbedder(
                **config["input_embedder"],
            )
        self.recycling_embedder = RecyclingEmbedder(
            **config["recycling_embedder"],
        )
        if self.globals.is_multimer:
            self.template_embedder = TemplateEmbedderMultimer(
                template_config,
            )
        else:
            self.template_embedder = TemplateEmbedder(
                template_config,
            )

        self.extra_msa_embedder = ExtraMSAEmbedder(
            **extra_msa_config["extra_msa_embedder"],
        )
        self.extra_msa_stack = ExtraMSAStack(
            **extra_msa_config["extra_msa_stack"],
            is_multimer=self.globals.is_multimer,
        )
        self.evoformer = EvoformerStack(
            **config["evoformer_stack"],
            is_multimer=self.globals.is_multimer,
        )
        self.structure_module = StructureModule(
            **config["structure_module"],
            is_multimer=self.globals.is_multimer,
        )

        self.aux_heads = AuxiliaryHeads(
            config["heads"],
        )

        self.config = config

    def iteration(
        self,
        feats: Dict[str, torch.Tensor],
        m_1_prev: torch.Tensor,
        z_prev: torch.Tensor,
        x_prev: torch.Tensor,
        _recycle: bool = True,
    ):

        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # Initialize the MSA and pair representations

        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = (
            self.input_embedder(
                feats["target_feat"],
                feats["residue_index"],
                feats["msa_feat"],
            )
            if not self.globals.is_multimer
            else self.input_embedder(feats)
        )

        # Initialize the recycling embeddings, if needs be
        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        x_prev, _ = pseudo_beta_fn(feats["aatype"], x_prev, None)
        x_prev = x_prev.to(dtype=z.dtype)

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
        )

        # If the number of recycling iterations is 0, skip recycling
        # altogether. We zero them this way instead of computing them
        # conditionally to avoid leaving parameters unused, which has annoying
        # implications for DDP training.
        # EDIT: This has since been removed from the official codebase (2cd61a)
        #        if(not _recycle):
        #            m_1_prev_emb *= 0
        #            z_prev_emb *= 0

        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        z += z_prev_emb

        if not self.training:
            # Possibly prevents memory fragmentation
            del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb

        # Embed the templates + merge with MSA/pair embeddings
        if self.config.template.enabled:
            template_feats = {
                k: v for k, v in feats.items() if k.startswith("template_")
            }

            if self.globals.is_multimer:
                asym_id = feats["asym_id"]
                multichain_mask_2d = asym_id[..., None] == asym_id[..., None, :]
                template_embeds = self.template_embedder(
                    template_feats,
                    z,
                    pair_mask.to(dtype=z.dtype),
                    no_batch_dims,
                    chunk_size=self.globals.chunk_size,
                    multichain_mask_2d=multichain_mask_2d,
                )
                feats["template_torsion_angles_mask"] = template_embeds["template_mask"]
            else:
                template_embeds = self.template_embedder(
                    template_feats,
                    z,
                    pair_mask.to(dtype=z.dtype),
                    no_batch_dims,
                    self.globals.chunk_size,
                )

            # [*, N, N, C_z]
            z = z + template_embeds["template_pair_embedding"]

            if self.config.template.embed_angles or (
                self.globals.is_multimer and self.config.template.enabled
            ):
                # [*, S = S_c + S_t, N, C_m]
                m = torch.cat([m, template_embeds["template_single_embedding"]], dim=-3)

                # [*, S, N]
                if not self.globals.is_multimer:
                    torsion_angles_mask = feats["template_torsion_angles_mask"]
                    msa_mask = torch.cat(
                        [feats["msa_mask"], torsion_angles_mask[..., 2]], dim=-2
                    )
                else:
                    msa_mask = torch.cat(
                        [feats["msa_mask"], template_embeds["template_mask"]],
                        dim=-2,
                    )

        # Embed extra MSA features + merge with pairwise embeddings
        if self.config.extra_msa.enabled:
            if self.globals.is_multimer:
                extra_msa_fn = data_transforms_multimer.build_extra_msa_feat
            else:
                extra_msa_fn = build_extra_msa_feat

            # [*, S_e, N, C_e]
            extra_msa_feat = extra_msa_fn(feats)
            extra_msa_feat = self.extra_msa_embedder(extra_msa_feat)

            # [*, N, N, C_z]
            z = self.extra_msa_stack(
                extra_msa_feat,
                z,
                msa_mask=feats["extra_msa_mask"].to(dtype=extra_msa_feat.dtype),
                chunk_size=self.globals.chunk_size,
                pair_mask=pair_mask.to(dtype=z.dtype),
                _mask_trans=self.config._mask_trans,
            )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]
        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            _mask_trans=self.config._mask_trans,
        )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            s,
            z,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, N, C_z]
        z_prev = z

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev

    def _disable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = None
        self.evoformer.blocks_per_ckpt = None

        for b in self.extra_msa_stack.blocks:
            b.ckpt = False

    def _enable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = (
            self.config.template.template_pair_stack.blocks_per_ckpt
        )
        self.evoformer.blocks_per_ckpt = self.config.evoformer_stack.blocks_per_ckpt

        for b in self.extra_msa_stack.blocks:
            b.ckpt = self.config.extra_msa.extra_msa_stack.ckpt

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None

        # Disable activation checkpointing for the first few recycling iters
        is_grad_enabled = torch.is_grad_enabled()
        self._disable_activation_checkpointing()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    self._enable_activation_checkpointing()
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats, m_1_prev, z_prev, x_prev, _recycle=(num_iters > 1)
                )

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        return outputs
