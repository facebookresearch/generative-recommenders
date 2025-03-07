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

#include "cute/tensor.hpp"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "seqlen.h"

namespace flash {

using namespace cute;

template <
    class TileShape_MK_,
    class Element,
    class ElementAccum,
    class ArchTag_,
    bool Clear_dQaccum,
    bool Jagged>
class FlashAttnBwdPreprocess {
 public:
  // Type Aliases
  using TileShape_MK = TileShape_MK_;
  using ArchTag = ArchTag_;

  static_assert(
      std::is_same_v<Element, cutlass::half_t> &&
          ArchTag::kMinComputeCapability >= 75 ||
      std::is_same_v<Element, cutlass::bfloat16_t> &&
          ArchTag::kMinComputeCapability >= 80 ||
      std::is_same_v<Element, cutlass::float_e4m3_t> &&
          ArchTag::kMinComputeCapability >= 89);

  static constexpr uint32_t MaxThreadsPerBlock = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
  static constexpr int SharedStorageSize = 0;

  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(
      get<1>(TileShape_MK{}) % kGmemElemsPerLoad == 0,
      "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBlockM = get<0>(TileShape_MK{});
  static constexpr int kHeadDim = get<1>(TileShape_MK{});
  // We want kBlockKGmem to be a power of 2 so that when we do the summing,
  // it's just between threads in the same warp
  static constexpr int kBlockKGmem =
      kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(
      MaxThreadsPerBlock % kGmemThreadsPerRow == 0,
      "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<
      Shape<
          Int<MaxThreadsPerBlock / kGmemThreadsPerRow>,
          Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{})); // Val layout, 8 or 16 vals
                                                     // per load

  static constexpr int kGmemElemsPerLoadAccum =
      sizeof(cute::uint128_t) / sizeof(ElementAccum);
  static_assert(
      (kBlockM * kHeadDim / kGmemElemsPerLoadAccum) % MaxThreadsPerBlock == 0,
      "MaxThreadsPerBlock must divide kBlockM * kHeadDim / kGmemElemsPerLoadAccum");
  using GmemLayoutAtomAccum = Layout<Shape<Int<MaxThreadsPerBlock>>>;
  using GmemTiledCopyAccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      GmemLayoutAtomAccum{},
      Layout<Shape<Int<kGmemElemsPerLoadAccum>>>{})); // Val layout, 4 vals per
                                                      // store

  using ShapeO =
      cute::Shape<int32_t, int32_t, int32_t, int32_t>; // (seqlen_q, d, head,
                                                       // batch)
  using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using ShapedQaccum =
      cute::Shape<int32_t, int32_t, int32_t>; // (seqlen_q * d, head, batch)
  using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

  // Device side arguments
  struct Arguments {
    ElementAccum* ptr_dQaccum;
    ShapedQaccum const shape_dQaccum;
    StridedQaccum const stride_dQaccum;
    int num_batch; // We need this to know the size of dq_semaphore in case of
                   // jagged
    int num_heads;
    int max_seq_len;
    int* dq_semaphore;
    int const* seq_offsets = nullptr;
  };

  // Kernel entry point API
  struct Params {
    ElementAccum* ptr_dQaccum;
    ShapedQaccum const shape_dQaccum;
    StridedQaccum const stride_dQaccum;
    int num_batch;
    int num_heads;
    int max_seq_len;
    int* dq_semaphore;
    int const* seq_offsets = nullptr;
  };

  // Convert to underlying arguments. In this case, a simple copy for the
  // aliased type.
  static Params to_underlying_arguments(Arguments const& args) {
    return {
        args.ptr_dQaccum,
        args.shape_dQaccum,
        args.stride_dQaccum,
        args.num_batch,
        args.num_heads,
        args.max_seq_len,
        args.dq_semaphore,
        args.seq_offsets};
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, [[maybe_unused]] char* smem_buf) {
    static constexpr int kBlockM = get<0>(TileShape_MK{});

    int const thread_idx = threadIdx.x;
    int const m_block = blockIdx.x;
    int const bidh = blockIdx.y;
    int const bidb = blockIdx.z;

    flash::SeqlenInfo<Jagged, kBlockM> seqlen_info(
        bidb, params.max_seq_len, params.seq_offsets);
    int const seqlen_o = seqlen_info.seqlen;
    if (Jagged && m_block * kBlockM >= seqlen_o) {
      return;
    }

    if constexpr (Clear_dQaccum) {
      Tensor mdQaccum = make_tensor(
          make_gmem_ptr(params.ptr_dQaccum),
          params.shape_dQaccum,
          params.stride_dQaccum)(_, bidh, !Jagged ? bidb : 0);
      Tensor gdQaccum = local_tile(
          cute::domain_offset(
              make_coord(seqlen_info.offset_padded * kHeadDim), mdQaccum),
          Shape<Int<kBlockM * kHeadDim>>{},
          make_coord(m_block));
      GmemTiledCopyAccum gmem_tiled_copy_dQaccum;
      auto gmem_thr_copy_dQaccum =
          gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
      Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
      Tensor zero = make_fragment_like(tdQgdQaccum);
      clear(zero);
      cute::copy(
          Copy_Atom<
              AutoVectorizingCopyWithAssumedAlignment<128>,
              ElementAccum>{},
          zero,
          tdQgdQaccum);
    }

    if (params.dq_semaphore != nullptr && thread_idx == 0) {
      int const num_batch = params.num_batch;
      int const num_head = params.num_heads;
      params.dq_semaphore
          [bidh + bidb * num_head + m_block * num_head * num_batch] = 0;
    }
  }
};

} // namespace flash
