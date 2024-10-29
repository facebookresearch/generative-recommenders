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

static constexpr int32_t kMaxThreads = 1024;

namespace gr {

template <typename index_t, typename val_t>
__global__
__launch_bounds__(kMaxThreads) void expand_1d_jagged_to_dense_cuda_kernel_(
    int64_t B,
    int64_t max_len,
    const at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits> values,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<val_t, 2, at::RestrictPtrTraits> output) {
  int64_t b = blockIdx.y;
  int64_t begin = offsets[b];
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t end = offsets[b + 1];
  if (end - begin == 0) {
    if (i < max_len) {
      output[b][i] = 0;
    }
  } else {
    if (i < std::min(end - begin, max_len)) {
      output[b][i] = values[i + begin];
    } else if (i < max_len) {
      output[b][i] = values[end - 1];
    }
  }
}

at::Tensor expand_1d_jagged_to_dense_cuda(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const int64_t max_len) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());
  TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(offsets.device().type() == at::DeviceType::CUDA);
  TORCH_CHECK(values.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(values.get_device() == offsets.get_device());
  TORCH_CHECK(max_len > 0);
  auto B = offsets.size(0) - 1;
  auto output = at::empty({B, max_len}, values.options());
  if (values.numel() == 0) {
    return output;
  }
  uint32_t nthreads_per_block = max_len > 64 ? 64 : max_len;
  dim3 grid_size = dim3(div_round_up(max_len, nthreads_per_block), B);
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      values.scalar_type(),
      "expand_1d_jagged_to_dense_cuda_input1",
      [&] {
        using val_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(
            offsets.scalar_type(),
            "expand_1d_jagged_to_dense_cuda_input2",
            [&] {
              using index_t = scalar_t;
              expand_1d_jagged_to_dense_cuda_kernel_<index_t, val_t><<<
                  grid_size,
                  nthreads_per_block,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  B,
                  max_len,
                  values.packed_accessor32<val_t, 1, at::RestrictPtrTraits>(),
                  offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  output.packed_accessor32<val_t, 2, at::RestrictPtrTraits>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });

  return output;
}

} // namespace gr
