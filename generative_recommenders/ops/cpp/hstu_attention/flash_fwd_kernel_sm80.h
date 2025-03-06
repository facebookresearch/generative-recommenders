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
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/numeric_types.h>

#include "seqlen.h"
#include "tile_scheduler.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class FlashAttnFwdSm80 {
 public:
  // Type Aliases
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
  static constexpr bool Transpose_V = CollectiveMainloop::Transpose_V;
  static constexpr int NumProducerThreads =
      CollectiveMainloop::NumProducerThreads;
  using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;

  // Mainloop derived types
  using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 80);

  using TileScheduler = TileScheduler_;
  using TileSchedulerArguments = typename flash::TileSchedulerArguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t NumThreads = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MaxThreadsPerBlock =
      CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor =
      NumThreads == 128 ? 2 : 1;

  // Kernel level shared memory storage
  // We overlap the shared memory for the mainloop and epilogue. However, we
  // only want smem_o to overlap with smem_v + smem_k and not smem_q and nothing
  // else, so we'll pad in case sizeof(smem_o) > sizeof(smem_v) +
  // sizeof(smem_k).
  static constexpr int mainloop_smem_padding_ =
      int(sizeof(typename CollectiveEpilogue::TensorStorage)) -
      int(sizeof(
          decltype((typename CollectiveMainloop::TensorStorage{}).smem_v))) -
      int(sizeof(
          decltype((typename CollectiveMainloop::TensorStorage{}).smem_k)));
  static constexpr int mainloop_smem_padding =
      mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      union {
        struct {
          cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)>
              padding_;
          typename CollectiveMainloop::TensorStorage mainloop;
        };
        // We want smem_o to line up with the start of smem_v
        typename CollectiveEpilogue::TensorStorage epilogue;
      };
    } tensors;

    alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the
  // aliased type.
  static Params to_underlying_arguments(Arguments const& args) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST(
        "to_underlying_arguments(): Setting persistent grid SM count to "
        << sm_count);

    cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
    return {
        CollectiveMainloop::to_underlying_arguments(args.mainloop),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue),
        hw_info,
        TileScheduler::to_underlying_arguments(args.scheduler)};
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(
        params.scheduler, params.hw_info.sm_count * MinBlocksPerMultiprocessor);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    TileScheduler scheduler(
        reinterpret_cast<typename TileScheduler::SharedStorage*>(
            &shared_storage.smem_scheduler));
    // Initialize matmul objects.
    TiledMma tiled_mma;

    scheduler.init_consumer();

    int warp_idx = cutlass::canonical_warp_idx_sync();
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = warp_idx == 0
             ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(
                   params.scheduler)
             : scheduler.template get_initial_work</*IsProducerWarp=*/false>(
                   params.scheduler);
         work_tile_info.is_valid(params.scheduler);
         work_tile_info = warp_idx == 0
             ? scheduler.template get_next_work</*IsProducerWarp=*/true>(
                   params.scheduler, work_tile_info)
             : scheduler.template get_next_work</*IsProducerWarp=*/false>(
                   params.scheduler, work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO =
          partition_fragment_C(tiled_mma, select<0, 2>(TileShape_MNK{}));
      // If there's tanh softcap, the scaling will be done before tanh.
      auto block_coord = work_tile_info.get_block_coord(params.scheduler);
      int const bidb = get<2>(block_coord);
      if constexpr (Is_FP8) {
        int const bidh = get<1>(block_coord);
        int const bidh_kv = bidh;
        float const q_descale = params.mainloop.ptr_q_descale == nullptr
            ? 1.0f
            : params.mainloop.ptr_q_descale
                  [bidb * get<0>(params.mainloop.stride_q_descale) +
                   bidh_kv * get<1>(params.mainloop.stride_q_descale)];
        float const k_descale = params.mainloop.ptr_k_descale == nullptr
            ? 1.0f
            : params.mainloop.ptr_k_descale
                  [bidb * get<0>(params.mainloop.stride_k_descale) +
                   bidh_kv * get<1>(params.mainloop.stride_k_descale)];
      }
      SeqlenInfo_t seqlen_info{
          bidb,
          get<0>(params.mainloop.shape_Q),
          params.mainloop.seq_offsets,
          params.mainloop.num_targets,
      };
      bool tile_valid = collective_mainloop.mma(
          params.mainloop,
          tOrO,
          threadIdx.x,
          seqlen_info,
          block_coord,
          shared_storage);
      scheduler.prefetch_next_work(params.scheduler, work_tile_info);
      if (tile_valid) {
        // if (threadIdx.x == 128) { printf("Before epilogue, bid.x = %d, bid.y
        // = %d, bid.z = %d, m_block = %d, bidb = %d, split_idx = %d\n",
        // blockIdx.x, blockIdx.y, blockIdx.z, m_block, bidb, split_idx); }
        collective_epilogue.store(
            params.epilogue,
            tOrO,
            shared_storage,
            tiled_mma,
            threadIdx.x,
            block_coord);
      } else {
        // Write 0 to gO and -inf to gLSE.
        // If Split, we don't have to write 0 to O if the mha_combine kernel is
        // used, since it will not use the value of O if LSE is -inf.
        collective_epilogue.template store_zero<true /*Clear_O*/>(
            params.epilogue, threadIdx.x, block_coord);
      }
    }
  }
};

} // namespace flash
