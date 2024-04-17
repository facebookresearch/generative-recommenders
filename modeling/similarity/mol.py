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

"""
Implements MoL (Mixture-of-Logits) in 
Revisiting Neural Retrieval on Accelerators (https://arxiv.org/abs/2306.04039, KDD'23).
"""
from typing import Callable, Dict, List, Optional, Tuple

import math

import torch
import torch.nn.functional as F

from modeling.initialization import init_mlp_xavier_weights_zero_bias


class SoftmaxDropout(torch.nn.Module):

    def __init__(
        self,
        dropout_rate: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self._softmax: torch.nn.Module = torch.nn.Softmax(dim=-1)
        self._dropout: torch.nn.Module = torch.nn.Dropout(p=dropout_rate)
        self._eps = eps

    def forward(self, x: torch.Tensor, tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tau is not None:
            x = x / tau
        x = self._dropout(self._softmax(x))
        return x / torch.clamp(x.sum(-1, keepdims=True), min=self._eps)


class SoftmaxDropoutCombiner(torch.nn.Module):

    def __init__(
        self,
        dropout_rate: float,
        eps: float,
        keep_debug_info: bool = False,
    ) -> None:
        super().__init__()

        self._softmax_dropout: torch.nn.Module = SoftmaxDropout(dropout_rate=dropout_rate, eps=eps)
        self._keep_debug_info: bool = keep_debug_info

    def forward(
        self,
        gating_weights: torch.Tensor,
        x: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        combined_logits = (self._softmax_dropout(gating_weights, tau) * x).sum(-1)
        if self._keep_debug_info:
            return combined_logits, {
                "gating_weights": gating_weights.detach().clone(),
                "x": x.detach().clone(),
            }
        else:
            return combined_logits


class IdentityMLPProjectionFn(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_num_features: int,
        output_dim: int,
        input_dropout_rate: float,
    ) -> None:
        super().__init__()

        self._output_num_features = output_num_features
        self._output_dim = output_dim
        if output_num_features > 1:
            self._proj_mlp = torch.nn.Sequential(
                torch.nn.Dropout(p=input_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=(output_num_features - 1) * output_dim,
                )
            ).apply(init_mlp_xavier_weights_zero_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_emb_0 = x[..., :self._output_dim]  # [.., D] -> [.., 1, D']
        if self._output_num_features > 1:
            return torch.cat([output_emb_0, self._proj_mlp(x)], dim=-1)
        return output_emb_0


class TauFn(torch.nn.Module):

    def __init__(
        self,
        alpha: float,
        item_sideinfo_dim: float,
    ) -> None:
        super().__init__()

        self._tau_fn: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(in_features=item_sideinfo_dim, out_features=1),
            torch.nn.Sigmoid(),
        )
        self._alpha: float = alpha

    def forward(
        self,
        item_sideinfo: torch.Tensor,
    ) -> torch.Tensor:
        return (self._tau_fn(item_sideinfo) + self._alpha) / self._alpha


class GeGLU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._w = torch.nn.Parameter(
            torch.empty((in_features, out_features * 2)).normal_(mean=0, std=0.02),
        )
        self._b = torch.nn.Parameter(
            torch.zeros((1, out_features * 2,)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size()[:-1]
        lhs, rhs = torch.split(
            torch.mm(x.reshape(-1, self._in_features), self._w) + self._b,
            [self._out_features, self._out_features],
            dim=-1,
        )
        return (F.gelu(lhs) * rhs).reshape(bs + (self._out_features,))


class SwiGLU(torch.nn.Module):
    """
    SwiGLU as proposed in ``GLU Variants Improve Transformer'' (https://arxiv.org/abs/2002.05202).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._w = torch.nn.Parameter(
            torch.empty((in_features, out_features * 2)).normal_(mean=0, std=0.02),
        )
        self._b = torch.nn.Parameter(
            torch.zeros((1, out_features * 2,)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size()[:-1]
        lhs, rhs = torch.split(
            torch.mm(x.reshape(-1, self._in_features), self._w) + self._b,
            [self._out_features, self._out_features],
            dim=-1,
        )
        return (F.silu(lhs) * rhs).reshape(bs + (self._out_features,))


class MoLGatingFn(torch.nn.Module):

    def __init__(
        self,
        num_logits: int,
        context_embedding_dim: int,
        item_embedding_dim: int,
        item_sideinfo_dim: int,
        context_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        item_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        ci_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        combination_type: str,
        normalization_fn: Callable[[int], torch.nn.Module],
        combine_item_sideinfo_into_ci: bool = False,
        gating_use_custom_tau: bool = False,
        gating_tau_alpha: float = 0.01,
    ) -> None:
        super().__init__()

        self._context_only_partial_module: Optional[torch.nn.Module] = (
            context_only_partial_fn(context_embedding_dim, num_logits)
            if context_only_partial_fn else None
        )
        self._item_only_partial_module: Optional[torch.nn.Module] = (
            item_only_partial_fn(item_embedding_dim + item_sideinfo_dim, num_logits)
            if item_only_partial_fn else None
        )
        self._ci_partial_module: Optional[torch.nn.Module] = (
            ci_partial_fn(
                num_logits +
                (item_sideinfo_dim if combine_item_sideinfo_into_ci else 0),
                num_logits,
            ) if ci_partial_fn is not None else None
        )
        if self._context_only_partial_module is None and self._item_only_partial_module is None and self._ci_partial_module is None:
            raise ValueError(
                "At least one of context_only_partial_fn, item_only_partial_fn, "
                "and ci_partial_fn must not be None."
            )
        self._num_logits: int = num_logits
        self._combination_type: str = combination_type
        self._combine_item_sideinfo_into_ci: bool = combine_item_sideinfo_into_ci
        self._normalization_fn: torch.nn.Module = normalization_fn(num_logits)
        if gating_use_custom_tau:
            self._tau_fn = TauFn(item_sideinfo_dim=item_sideinfo_dim, alpha=gating_tau_alpha)
        else:
            self._tau_fn = None

    def forward(
        self,
        logits: torch.Tensor,
        context_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_sideinfo: Optional[torch.Tensor] = None,
        batch_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits: (B, X, L) x float
            context_embeddings: (B, D) x float
            item_embeddings: (1/B, X, D') x float
            item_sideinfo: (1/B, X, F) x float or None
            batch_id: if present, (,) x int

        Returns:
            (B, X) x float
        """
        B, X, _ = logits.size()
        # [B, 1, F], [1/B, X, F], [B, X, F]
        context_partial_inputs, item_partial_inputs, ci_partial_inputs = None, None, None
        if self._context_only_partial_module is not None:
            context_partial_inputs = (
                self._context_only_partial_module(context_embeddings).unsqueeze(1)
            )
        if self._item_only_partial_module is not None:
            if item_sideinfo is not None:
                item_embeddings = torch.cat([item_embeddings, item_sideinfo], dim=-1)
            item_partial_inputs = self._item_only_partial_module(item_embeddings)
        if self._ci_partial_module is not None:
            if self._combine_item_sideinfo_into_ci:
                B_prime = item_sideinfo.size(0)
                if B_prime == 1:
                    item_sideinfo = item_sideinfo.expand(B, -1, -1)
                ci_partial_inputs = self._ci_partial_module(
                    torch.cat([logits, item_sideinfo], dim=2)
                )
            else:
                ci_partial_inputs = self._ci_partial_module(logits)

        if self._combination_type == "glu_silu":
            gating_inputs = context_partial_inputs * item_partial_inputs + ci_partial_inputs
            gating_weights = gating_inputs * F.sigmoid(gating_inputs)
        elif self._combination_type == "glu_silu_ln":
            gating_inputs = context_partial_inputs * item_partial_inputs + ci_partial_inputs
            gating_weights = (
                gating_inputs
                * F.sigmoid(F.layer_norm(gating_inputs, normalized_shapes=[self._num_logits]))
            )
        elif self._combination_type == "silu":
            if context_partial_inputs is not None:
                gating_inputs = context_partial_inputs.expand(-1, X, -1)
            else:
                gating_inputs = None

            if gating_inputs is None:
                gating_inputs = item_partial_inputs
            elif item_partial_inputs is not None:
                gating_inputs = gating_inputs + item_partial_inputs

            if gating_inputs is None:
                gating_inputs = ci_partial_inputs
            elif ci_partial_inputs is not None:
                gating_inputs = gating_inputs + ci_partial_inputs

            gating_weights = gating_inputs * F.sigmoid(gating_inputs)
        elif self._combination_type == "none":
            gating_inputs = context_partial_inputs
            if gating_inputs is None:
                gating_inputs = item_partial_inputs
            elif item_partial_inputs is not None:
                gating_inputs += item_partial_inputs
            if gating_inputs is None:
                gating_inputs = ci_partial_inputs
            elif ci_partial_inputs is not None:
                gating_inputs += ci_partial_inputs
            gating_weights = gating_inputs
        else:
            raise ValueError(f"Unknown combination_type {self._combination_type}")

        tau = None
        if self._tau_fn is not None:
            tau = self._tau_fn(item_sideinfo)
        return self._normalization_fn(gating_weights, logits, tau)  #, {}


class MoLSimilarity(torch.nn.Module):
    """
    Implements MoL (Mixture-of-Logits) learned similarity in 
    Revisiting Neural Retrieval on Accelerators (https://arxiv.org/abs/2306.04039, KDD'23).
    """
    def __init__(
        self,
        input_embedding_dim: int,
        item_embedding_dim: int,
        dot_product_dimension: int,
        input_dot_product_groups: int,
        item_dot_product_groups: int,
        temperature: float,
        dot_product_l2_norm: bool,
        num_precomputed_logits: int,
        item_sideinfo_dim: int,
        context_proj_fn: Callable[[int, int], torch.nn.Module],
        item_proj_fn: Callable[[int, int], torch.nn.Module],
        gating_context_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        gating_item_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        gating_ci_partial_fn: Optional[Callable[[int], torch.nn.Module]],
        gating_combination_type: str,
        gating_normalization_fn: Callable[[int], torch.nn.Module],
        eps: float,
        gating_combine_item_sideinfo_into_ci: bool = False,
        gating_use_custom_tau: bool = False,
        gating_tau_alpha: float = 0.01,
        bf16_training: bool = False,
    ) -> None:
        super().__init__()

        self._gating_fn: MoLGatingFn = MoLGatingFn(
            num_logits=input_dot_product_groups * item_dot_product_groups + num_precomputed_logits,
            context_embedding_dim=input_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            item_sideinfo_dim=item_sideinfo_dim,
            context_only_partial_fn=gating_context_only_partial_fn,
            item_only_partial_fn=gating_item_only_partial_fn,
            ci_partial_fn=gating_ci_partial_fn,
            combine_item_sideinfo_into_ci=gating_combine_item_sideinfo_into_ci,
            combination_type=gating_combination_type,
            normalization_fn=gating_normalization_fn,
            gating_use_custom_tau=gating_use_custom_tau,
            gating_tau_alpha=gating_tau_alpha,
        )
        self._context_proj_module: torch.nn.Module = context_proj_fn(
            input_embedding_dim, dot_product_dimension * input_dot_product_groups,
        )
        self._item_proj_module: torch.nn.Module = item_proj_fn(
            item_embedding_dim,  # + item_sideinfo_dim,
            dot_product_dimension * item_dot_product_groups,
        )
        self._item_sideinfo_dim: int = item_sideinfo_dim
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        self._input_dot_product_groups: int = input_dot_product_groups
        self._item_dot_product_groups: int = item_dot_product_groups
        self._dot_product_dimension: int = dot_product_dimension
        self._temperature: float = temperature
        self._eps: float = eps
        self._bf16_training: bool = bf16_training

    def _frequency_estimator_old(self, ids: torch.Tensor) -> torch.Tensor:
        ids_shape = ids.size()
        ids = ids.reshape(-1)
        temp = (
            (1 - self._lnx_estimator_alpha) * self._B[ids] +
            self._lnx_estimator_alpha * (
                self._lnx_num_batches + 1 - self._A[ids]
            )
        )
        temp = torch.clamp(temp, max=self._lnx_estimator_b_cap)
        if self.train:
            self._lnx_num_batches = self._lnx_num_batches + 1
            self._B[ids] = temp
            self._A[ids] = self._lnx_num_batches
        return 1.0 / temp.reshape(ids_shape)

    def _frequency_estimator(self, ids: torch.Tensor, update: bool) -> torch.Tensor:
        ids_shape = ids.size()
        ids = ids.reshape(-1)
        sorted_id_values, sorted_id_indices = ids.sort()
        (
            sorted_unique_ids,
            sorted_unique_inverses,
            sorted_unique_cnts,
        ) = sorted_id_values.unique_consecutive(
            return_counts=True,
            return_inverse=True,
        )
        most_recent_batches = torch.zeros_like(sorted_unique_ids, dtype=torch.int64)
        most_recent_batches[sorted_unique_inverses] = (
            sorted_id_indices + self._lnx_estimator_num_elements
        )
        delta_batches = torch.zeros_like(ids, dtype=torch.float32)
        delta_batches[sorted_id_indices] = torch.gather(
            input=(most_recent_batches - self._A[sorted_unique_ids]).float()
            / sorted_unique_cnts.float(),
            dim=0,
            index=sorted_unique_inverses,
        )

        temp = (1 - self._lnx_estimator_alpha) * self._B[ids] + self._lnx_estimator_alpha * delta_batches
        temp = torch.clamp(temp, max=self._lnx_estimator_b_cap)

        if update:
            self._B[ids] = temp
            self._A[sorted_unique_ids] = most_recent_batches
            self._lnx_estimator_num_elements = self._lnx_estimator_num_elements + ids.numel()
        return 1.0 / temp.reshape(ids_shape)

    def get_query_component_embeddings(
        self,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings: (B, self._input_embedding_dim,) x float.
        
        Returns:
            (B, query_dot_product_groups, dot_product_embedding_dim) x float.
        """
        with torch.autocast(enabled=self._bf16_training, dtype=torch.bfloat16, device_type='cuda'):
            split_user_embeddings = self._context_proj_module(input_embeddings).reshape(
                (input_embeddings.size(0), self._input_dot_product_groups, self._dot_product_dimension)
            )
            if self._dot_product_l2_norm:
                split_user_embeddings = split_user_embeddings / torch.clamp(
                    torch.linalg.norm(
                        split_user_embeddings, ord=None, dim=-1, keepdim=True,
                    ), min=self._eps,
                )
            return split_user_embeddings

    def get_item_component_embeddings(
        self,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings: (B, self._input_embedding_dim,) x float.
        
        Returns:
            (B, item_dot_product_groups, dot_product_embedding_dim) x float.
        """
        with torch.autocast(enabled=self._bf16_training, dtype=torch.bfloat16, device_type='cuda'):
            split_item_embeddings = self._item_proj_module(input_embeddings).reshape(
                input_embeddings.size()[:-1] + (self._item_dot_product_groups, self._dot_product_dimension,)
            )
            if self._dot_product_l2_norm:
                split_item_embeddings = split_item_embeddings / torch.clamp(
                    torch.linalg.norm(
                        split_item_embeddings, ord=None, dim=-1, keepdim=True,
                    ), min=self._eps,
                )
            return split_item_embeddings

    def forward(
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_sideinfo: Optional[torch.Tensor],
        item_ids: torch.Tensor,
        precomputed_logits: Optional[torch.Tensor] = None,
        batch_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, self._input_embedding_dim)
            item_embeddings: (1/B, X, self._item_embedding_dim)
            item_sideinfo: (1/B, X, self._item_sideinfo_dim)
            item_ids: (1/B, X,)
            precomputed_logits: (B, X, self._num_precomputed_logits,)
        """
        with torch.autocast(enabled=self._bf16_training, dtype=torch.bfloat16, device_type='cuda'):
            B = input_embeddings.size(0)
            B_prime, X, D = item_embeddings.shape

            #if self._item_sideinfo_dim > 0:
            #    item_proj_input = torch.cat([item_embeddings, item_sideinfo], dim=-1)
            #else:
            item_proj_input = item_embeddings

            split_user_embeddings = self._context_proj_module(input_embeddings).reshape(
                (B, self._input_dot_product_groups, self._dot_product_dimension)
            )
            split_item_embeddings = self._item_proj_module(item_proj_input).reshape(
                (B_prime, X, self._item_dot_product_groups, self._dot_product_dimension)
            )
            if self._dot_product_l2_norm:
                split_user_embeddings = split_user_embeddings / torch.clamp(
                    torch.linalg.norm(
                        split_user_embeddings, ord=None, dim=-1, keepdim=True,
                    ), min=self._eps,
                )
                split_item_embeddings = split_item_embeddings / torch.clamp(
                    torch.linalg.norm(
                        split_item_embeddings, ord=None, dim=-1, keepdim=True,
                    ), min=self._eps,
                )
            if B_prime == 1:
                #logits = torch.mm(split_user_embeddings, split_item_embeddings.t()).reshape(
                #    B, self._input_dot_product_groups, X, self._item_dot_product_groups
                #).permute(0, 2, 1, 3)  # (bn, xm) -> (b, n, x, m) -> (b, x, n, m)
                logits = torch.einsum(
                    "bnd,xmd->bxnm", split_user_embeddings, split_item_embeddings.squeeze(0)
                ).reshape(B, X, self._input_dot_product_groups * self._item_dot_product_groups)
            else:
                #logits = torch.bmm(
                #    split_user_embeddings,
                #    split_item_embeddings.permute(0, 2, 1)   # [b, n, d], [b, xm, d] -> [b, n, xm]
                #).reshape(B, self._input_dot_product_groups, X, self._item_dot_product_groups).permute(0, 2, 1, 3)
                logits = torch.einsum(
                    "bnd,bxmd->bxnm", split_user_embeddings, split_item_embeddings
                ).reshape(B, X, self._input_dot_product_groups * self._item_dot_product_groups)
            # [b, x, n, m] -> [b, x, n * m]
            #logits = logits.reshape(B, X, self._input_dot_product_groups * self._item_dot_product_groups)

            return self._gating_fn(
                logits=logits / self._temperature,  # [B, X, L]
                context_embeddings=input_embeddings,  # [B, D]
                item_embeddings=item_embeddings,  # [1/B, X, D']
                item_sideinfo=item_sideinfo,  # [1/B, X, D'']
                batch_id=batch_id,
            )