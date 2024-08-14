# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import gin
import torch

from modeling.initialization import init_mlp_xavier_weights_zero_bias
from modeling.similarity.dot_product import DotProductSimilarity
from modeling.similarity.mol import MoLSimilarity, GeGLU, SoftmaxDropoutCombiner, IdentityMLPProjectionFn


@gin.configurable
def create_mol_interaction_module(
    query_embedding_dim: int,
    item_embedding_dim: int,
    dot_product_dimension: int,
    query_dot_product_groups: int,
    item_dot_product_groups: int,
    temperature: float,
    dot_product_l2_norm: bool,
    query_use_identity_fn: bool,
    query_dropout_rate: float,
    query_hidden_dim: int,
    item_use_identity_fn: bool,
    item_dropout_rate: float,
    item_hidden_dim: int,
    gating_combination_type: str,
    gating_qi_hidden_dim: int,
    gating_query_hidden_dim: int,
    gating_item_hidden_dim: int,
    gating_softmax_dropout_rate: float,
    bf16_training: bool,
    gating_query_fn: bool = True,
    gating_item_fn: bool = True,
    gating_qi_dropout_rate: float = 0.0,
    gating_item_dropout_rate: float = 0.0,
    gating_use_custom_tau: bool = False,
    gating_tau_alpha: float = 0.01,
    eps: float = 1e-6,
) -> Tuple[MoLSimilarity, str]:
    mol_module = MoLSimilarity(
        input_embedding_dim=query_embedding_dim,
        item_embedding_dim=item_embedding_dim,
        dot_product_dimension=dot_product_dimension,
        input_dot_product_groups=query_dot_product_groups,
        item_dot_product_groups=item_dot_product_groups,
        temperature=temperature,
        dot_product_l2_norm=dot_product_l2_norm,
        num_precomputed_logits=0,
        # item_feature_embedding_dim * 3 if not ablate_item_features else
        item_sideinfo_dim=0,  # not configured
        context_proj_fn=lambda input_dim, output_dim: IdentityMLPProjectionFn(
            input_dim=input_dim,
            output_num_features=query_dot_product_groups,
            output_dim=output_dim // query_dot_product_groups,
            input_dropout_rate=query_dropout_rate,
        ) if query_use_identity_fn else
            (
                torch.nn.Sequential(
                    torch.nn.Dropout(p=query_dropout_rate),
                    GeGLU(in_features=input_dim, out_features=query_hidden_dim,),
                    torch.nn.Linear(
                        in_features=query_hidden_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias) if query_hidden_dim > 0 else torch.nn.Sequential(
                    torch.nn.Dropout(p=query_dropout_rate),
                    torch.nn.Linear(
                        in_features=input_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            ),
        item_proj_fn=lambda input_dim, output_dim: IdentityMLPProjectionFn(
            input_dim=input_dim,
            output_num_features=item_dot_product_groups,
            output_dim=output_dim // item_dot_product_groups,
            input_dropout_rate=item_dropout_rate,
        ) if item_use_identity_fn else
            (
                torch.nn.Sequential(
                    torch.nn.Dropout(p=item_dropout_rate),
                    GeGLU(in_features=input_dim, out_features=item_hidden_dim,),
                    torch.nn.Linear(
                        in_features=item_hidden_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias) if item_hidden_dim > 0 else
                torch.nn.Sequential(
                    torch.nn.Dropout(p=item_dropout_rate),
                    torch.nn.Linear(
                        in_features=input_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            ),
        gating_context_only_partial_fn=lambda input_dim, output_dim: torch.nn.Sequential(
            torch.nn.Linear(
                in_features=input_dim,
                out_features=gating_query_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=gating_query_hidden_dim,
                out_features=output_dim,
                bias=False,
            ),
        ).apply(init_mlp_xavier_weights_zero_bias) if gating_query_fn else None,
        gating_item_only_partial_fn=lambda input_dim, output_dim: torch.nn.Sequential(
            torch.nn.Dropout(p=gating_item_dropout_rate),
            torch.nn.Linear(
                in_features=input_dim,
                out_features=gating_item_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=gating_item_hidden_dim,
                out_features=output_dim,
                bias=False,
            ),
        ).apply(init_mlp_xavier_weights_zero_bias) if gating_item_fn else None,
        gating_ci_partial_fn=lambda input_dim, output_dim: torch.nn.Sequential(
            torch.nn.Dropout(p=gating_qi_dropout_rate),
            torch.nn.Linear(
                in_features=input_dim,
                out_features=gating_qi_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=gating_qi_hidden_dim,
                out_features=output_dim,
            ),
        ).apply(init_mlp_xavier_weights_zero_bias) if gating_qi_hidden_dim > 0 else torch.nn.Sequential(
            torch.nn.Dropout(p=gating_qi_dropout_rate),
            torch.nn.Linear(
                in_features=input_dim,
                out_features=output_dim,
            ),
        ).apply(init_mlp_xavier_weights_zero_bias),
        gating_combination_type=gating_combination_type,
        gating_normalization_fn=lambda _: SoftmaxDropoutCombiner(dropout_rate=gating_softmax_dropout_rate, eps=1e-6),
        eps=eps,
        gating_combine_item_sideinfo_into_ci=False,
        gating_use_custom_tau=gating_use_custom_tau,
        gating_tau_alpha=gating_tau_alpha,
        bf16_training=bf16_training,
    )
    interaction_module_debug_str = (
        f"MoL-{query_dot_product_groups}x{item_dot_product_groups}x{dot_product_dimension}"
        + f"-t{temperature}-d{gating_softmax_dropout_rate}"
        + f"{'-l2' if dot_product_l2_norm else ''}"
        + (f"-q{query_hidden_dim}d{query_dropout_rate}geglu" if query_hidden_dim > 0 else f"-qd{query_dropout_rate}")
        + (
            "-i_id" if item_use_identity_fn else
            (f"-{item_hidden_dim}d{item_dropout_rate}-geglu" if item_hidden_dim > 0 else f"-id{item_dropout_rate}")
        )
        + (f"-gq{gating_query_hidden_dim}" if gating_query_fn else "")
        + (f"-gi{gating_item_hidden_dim}d{gating_item_dropout_rate}" if gating_item_fn else "")
        + f"-gqi{gating_qi_hidden_dim}d{gating_qi_dropout_rate}-x-{gating_combination_type}"
    )
    if gating_use_custom_tau:
        interaction_module_debug_str += f"-tau{gating_tau_alpha}"
    return mol_module, interaction_module_debug_str


@gin.configurable
def get_similarity_function(
    module_type: str,
    query_embedding_dim: int,
    item_embedding_dim: int,
    bf16_training: bool = False,
    activation_checkpoint: bool = False,
) -> Tuple[torch.nn.Module, str]:
    if module_type == "DotProduct":
        interaction_module = DotProductSimilarity()
        interaction_module_debug_str = "DotProduct"
    elif module_type == "MoL":
        interaction_module, interaction_module_debug_str = create_mol_interaction_module(
            query_embedding_dim=query_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            bf16_training=bf16_training,
        )
    else:
        raise ValueError(f"Unknown interaction_module_type {module_type}")
    return interaction_module, interaction_module_debug_str