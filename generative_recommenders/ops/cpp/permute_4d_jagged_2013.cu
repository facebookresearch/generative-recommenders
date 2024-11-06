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

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "common.h"
#include "fbgemm_gpu/sparse_ops.h" // @manual
#include "fbgemm_gpu/utils/fixed_divisor.cuh" // @manual

namespace gr {

static constexpr int32_t kMaxThreads = 1024;

template <typename index_t>
__global__
__launch_bounds__(kMaxThreads) void _permute_4d_jagged_2013_lengths_cuda_kernel(
    int32_t D0_D1_D2,
    int32_t D0,
    int32_t D1,
    int32_t D2,
    fbgemm_gpu::FixedDivisor fd_d0d1,
    fbgemm_gpu::FixedDivisor fd_d1,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> lengths,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        permuted_lengths) {
  int32_t idx_start = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t stride = blockDim.x * gridDim.x;
  for (int32_t output_idx = idx_start; output_idx < D0_D1_D2;
       output_idx += stride) {
    int32_t d2;
    int32_t d0d1;
    int32_t d1;
    int32_t d0;
    fd_d0d1.DivMod(output_idx, &d2, &d0d1);
    fd_d1.DivMod(d0d1, &d0, &d1);
    auto input_idx = d0 * D1 * D2 + d1 * D2 + d2;
    permuted_lengths[output_idx] = lengths[input_idx];
  }
}

template <typename index_t, typename val_t>
__global__
__launch_bounds__(kMaxThreads) void _permute_4d_jagged_2013_values_cuda_kernel(
    int32_t L,
    int32_t D0_D1_D2,
    int32_t D0,
    int32_t D1,
    int32_t D2,
    fbgemm_gpu::FixedDivisor fd_d0d1,
    fbgemm_gpu::FixedDivisor fd_d1,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        permuted_offsets,
    const at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits> values,
    at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits>
        permuted_values) {
  int32_t idx_start = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t stride = gridDim.x * blockDim.y;
  for (auto output_idx = idx_start; output_idx < D0_D1_D2; output_idx += stride) {
    auto output_start = permuted_offsets[output_idx];
    int32_t segment_length;
    if (output_idx == D0_D1_D2 - 1) {
      segment_length = L - output_start;
    } else {
      segment_length = permuted_offsets[output_idx + 1] - output_start;
    }
    int32_t d2;
    int32_t d0d1;
    int32_t d1;
    int32_t d0;
    fd_d0d1.DivMod(output_idx, &d2, &d0d1);
    fd_d1.DivMod(d0d1, &d0, &d1);
    int32_t input_idx = d0 * D1 * D2 + d1 * D2 + d2;
    auto input_start = offsets[input_idx];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_values[output_start + i] = values[input_start + i];
    }
  }
}

std::tuple<at::Tensor, at::Tensor> permute_4d_jagged_2013_cuda(
    const at::Tensor& lengths,
    const at::Tensor& values) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());
  TORCH_INTERNAL_ASSERT(lengths.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CUDA);
  TORCH_CHECK(values.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(values.get_device() == lengths.get_device());
  int32_t D0 = lengths.size(0);
  int32_t D1 = lengths.size(1);
  int32_t D2 = lengths.size(2);
  int32_t D0_D1_D2 = D0 * D1 * D2;
  TORCH_CHECK(D0_D1_D2 < std::numeric_limits<int32_t>::max());
  auto L = values.numel();
  auto input_lengths = lengths.view({D0_D1_D2});
  auto permuted_lengths = at::empty({D0_D1_D2}, lengths.options());
  auto permuted_values = at::empty(L, values.options());
  if (L == 0) {
    return {permuted_lengths, permuted_values};
  }
  auto fd_d0d1 = fbgemm_gpu::FixedDivisor(D0 * D1);
  auto fd_d1 = fbgemm_gpu::FixedDivisor(D1);
  uint32_t threads_1 = kMaxThreads;
  auto blocks_1 = div_round_up(D0_D1_D2, threads_1);
  AT_DISPATCH_INTEGRAL_TYPES(
      lengths.scalar_type(),
      "permute_4d_jagged_2013_lengths_cuda_kernel_",
      [&] {
        using index_t = scalar_t;
        _permute_4d_jagged_2013_lengths_cuda_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                D0_D1_D2,
                D0,
                D1,
                D2,
                fd_d0d1,
                fd_d1,
                input_lengths
                    .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                permuted_lengths
                    .packed_accessor32<index_t, 1, at::RestrictPtrTraits>());
      });
  auto offsets = fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(input_lengths);
  auto permuted_offsets =
      fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(permuted_lengths);
  uint32_t D_blocks = 32;
  dim3 threads_2(32, D_blocks);
  auto blocks_2 = div_round_up(D0_D1_D2, D_blocks);
  AT_DISPATCH_INTEGRAL_TYPES(
      lengths.scalar_type(),
      "permute_4d_jagged_2013_values_cuda_kernel_input1",
      [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            values.scalar_type(),
            "permute_4d_jagged_2013_values_cuda_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _permute_4d_jagged_2013_values_cuda_kernel<index_t, val_t><<<
                  blocks_2,
                  threads_2,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  L,
                  D0_D1_D2,
                  D0,
                  D1,
                  D2,
                  fd_d0d1,
                  fd_d1,
                  offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  permuted_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  values.packed_accessor32<val_t, 1, at::RestrictPtrTraits>(),
                  permuted_values
                      .packed_accessor32<val_t, 1, at::RestrictPtrTraits>());
            });
      });
  return {permuted_lengths.view({D2, D0, D1}), permuted_values};
}
} // namespace gr
