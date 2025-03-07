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

#pragma once

#include <tuple>

// Return {kBlockM, kBlockN, Mma1_is_RS}
constexpr std::tuple<int, int, bool> tile_size_fwd_sm90(
    int headdim,
    bool is_causal,
    bool is_local,
    int element_size = 2,
    bool v_colmajor = false) {
  if (element_size == 2) {
    if (headdim <= 64) {
      return {192, 128, true};
      // Good for long seqlen (>= 4k) but suffers from tile quantization at
      // short seqlen return {192, is_causal || is_local ? 192 : 176, true,
      // false};
    } else if (headdim <= 96) {
      return {192, is_local ? 128 : 144, false};
    } else if (headdim <= 128) {
      return {128, is_causal || is_local ? 128 : 176, true};
      // {128, 192, false, false} and {192, 128, false, true} are quite good too
      // 128 x 192 hits the limit of smem if Mma1_is_RS, 128 x 144 hits the
      // limit if !Mma1_is_RS
    } else if (headdim <= 192) {
      return {
          128, is_local ? 96 : 112, true}; // 128 x 112 hits the limit of smem
    } else {
      return {128, is_local ? 64 : 80, true}; // 128 x 80 hits the limit of smem
    }
  } else {
    if (headdim <= 64) {
      return {192, 160, true};
    } else if (headdim <= 96) {
      return {192, 128, true};
    } else if (headdim <= 128) {
      return {128, (v_colmajor ? 192 : 224), true};
    } else if (headdim <= 192) {
      return {128, 160, true};
    } else {
      return {128, is_local ? 64 : 128, true};
    }
  }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
    bool sm86_or_89,
    int headdim,
    bool is_causal,
    bool is_local,
    int element_size = 2) {
  if (element_size == 2) {
    if (headdim <= 64) {
      return {128, (is_local ? 96 : 112), 4, 1, false};
    } else if (headdim <= 96) {
      return {128, is_local ? 48 : 64, 4, 1, false};
    } else if (headdim <= 128) {
      bool const use_8_warps = sm86_or_89;
      return {
          128,
          use_8_warps ? (is_local ? 96 : 128) : (is_local ? 48 : 64),
          use_8_warps ? 8 : 4,
          1,
          use_8_warps};
    } else if (headdim <= 192) {
      bool const kBlockN_64 = is_local;
      return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
    } else {
      return {
          128,
          sm86_or_89 ? (is_local ? 48 : 64) : (is_local ? 64 : 96),
          8,
          1,
          false};
    }
  } else {
    // Placeholder for now
    return {128, 64, 8, 2, false};
  }
}
