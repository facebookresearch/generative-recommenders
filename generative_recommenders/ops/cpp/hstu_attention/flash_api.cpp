/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 *Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include <torch/version.h> // For TORCH_VERSION* macros

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>
#include <torch/library.h> // @manual
#include <torch/nn/functional.h>
#include <torch/torch.h> // @manual

#include "flash_common.h"

namespace hstu {

TORCH_LIBRARY_FRAGMENT(hstu, m) {
  m.def(
      "hstu_mha_fwd("
      "int max_seq_len, "
      "float alpha, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor? seq_offsets, "
      "bool causal, "
      "Tensor? num_targets, "
      "int max_attn_len, "
      "int min_full_attn_seq_len, "
      "int contextual_seq_len, "
      "Tensor? q_descale, "
      "Tensor? k_descale, "
      "Tensor? v_descale, "
      "int sm_margin"
      ") -> Tensor");

  m.def(
      "hstu_mha_bwd("
      "int max_seq_len, "
      "float alpha, "
      "Tensor dout, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor dq, "
      "Tensor dk, "
      "Tensor dv, "
      "Tensor? seq_offsets, "
      "bool causal, "
      "Tensor? num_targets, "
      "int max_attn_len, "
      "int min_full_attn_seq_len, "
      "int contextual_seq_len, "
      "bool sort_by_length,"
      "bool deterministic,"
      "int sm_margin"
      ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(hstu, CUDA, m) {
  m.impl(
      "hstu_mha_fwd",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(hstu_mha_fwd)));
  m.impl(
      "hstu_mha_bwd",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(hstu_mha_bwd)));
}
} // namespace hstu
