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

namespace hstu {

static constexpr int32_t kMaxThreads = 1024;

template <typename index_t, typename val_t>
__global__
__launch_bounds__(kMaxThreads) void _concat_1d_jagged_jagged_cuda_kernel(
    int32_t B,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets_left,
    const at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits>
        values_left,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets_right,
    const at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits>
        values_right,
    at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits>
        combined_values) {
  for (int32_t b = blockIdx.x * blockDim.y + threadIdx.y; b < B;
       b += gridDim.x * blockDim.y) {
    auto left_start = offsets_left[b];
    auto left_len = offsets_left[b + 1] - left_start;
    auto right_start = offsets_right[b];
    auto right_len = offsets_right[b + 1] - right_start;
    auto combined_start = left_start + right_start;
    for (int32_t i = threadIdx.x; i < left_len + right_len; i += blockDim.x) {
      if (i < left_len) {
        combined_values[combined_start + i] = values_left[left_start + i];
      } else {
        combined_values[combined_start + i] =
            values_right[right_start + i - left_len];
      }
    }
  }
}

at::Tensor concat_1d_jagged_jagged_cuda(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values_left.get_device());
  TORCH_INTERNAL_ASSERT(lengths_left.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(values_left.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(lengths_right.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(values_right.device().type() == at::DeviceType::CUDA);
  auto L = values_left.numel() + values_right.numel();
  TORCH_CHECK(L < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(values_left.get_device() == lengths_left.get_device());
  TORCH_CHECK(values_left.get_device() == lengths_right.get_device());
  TORCH_CHECK(values_left.get_device() == values_right.get_device());
  auto B = lengths_left.size(0);
  auto combined_values = at::empty({L}, values_left.options());
  if (L == 0) {
    return combined_values;
  }
  const auto offsets_left =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(lengths_left.view({-1}));
  const auto offsets_right =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(lengths_right.view({-1}));
  uint32_t B_blocks = 32;
  dim3 threads(32, B_blocks);
  auto blocks = div_round_up(B, B_blocks);
  AT_DISPATCH_INTEGRAL_TYPES(
      lengths_left.scalar_type(),
      "concat_1d_jagged_jagged_values_cuda_kernel_input1",
      [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            values_left.scalar_type(),
            "concat_1d_jagged_jagged_values_cuda_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _concat_1d_jagged_jagged_cuda_kernel<index_t, val_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      B,
                      offsets_left.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      values_left
                          .packed_accessor32<val_t, 1, at::RestrictPtrTraits>(),
                      offsets_right.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      values_right
                          .packed_accessor32<val_t, 1, at::RestrictPtrTraits>(),
                      combined_values.packed_accessor32<
                          val_t,
                          1,
                          at::RestrictPtrTraits>());
            });
      });
  return combined_values;
}
} // namespace hstu
