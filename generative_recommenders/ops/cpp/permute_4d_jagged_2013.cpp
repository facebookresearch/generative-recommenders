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
#include <torch/extension.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops.h" // @manual

namespace gr {

template <typename index_t>
void _permute_4d_jagged_2013_lengths_cpu_kernel(
    int64_t D0,
    int64_t D1,
    int64_t D2,
    const at::TensorAccessor<index_t, 3>& lengths,
    at::TensorAccessor<index_t, 3> permuted_lengths) {
  for (auto d2 : c10::irange(D2)) {
    for (auto d0 : c10::irange(D0)) {
      for (auto d1 : c10::irange(D1)) {
        permuted_lengths[d2][d0][d1] = lengths[d0][d1][d2];
      } // for each d1
    } // for each d0
  } // for each d2
}

template <typename index_t, typename val_t>
void _permute_4d_jagged_2013_values_cpu_kernel(
    int64_t D0,
    int64_t D1,
    int64_t D2,
    const at::TensorAccessor<index_t, 3>& lengths,
    const at::TensorAccessor<index_t, 3>& offsets,
    const at::TensorAccessor<index_t, 3>& permuted_offsets,
    const at::TensorAccessor<val_t, 1>& values,
    at::TensorAccessor<val_t, 1> permuted_values) {
  for (auto d2 : c10::irange(D2)) {
    for (auto d0 : c10::irange(D0)) {
      for (auto d1 : c10::irange(D1)) {
        auto len = lengths[d0][d1][d2];
        auto input_start = offsets[d0][d1][d2];
        auto output_start = permuted_offsets[d2][d0][d1];
        for (auto i : c10::irange(len)) {
          permuted_values[output_start + i] = values[input_start + i];
        } // for each i
      } // for each d1
    } // for each d0
  } // for each d2
}

std::tuple<at::Tensor, at::Tensor> permute_4d_jagged_2013_cpu(
    const at::Tensor& lengths,
    const at::Tensor& values) {
  TORCH_INTERNAL_ASSERT(lengths.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CPU);
  TORCH_CHECK(values.numel() < std::numeric_limits<int32_t>::max());
  auto D0 = lengths.size(0);
  auto D1 = lengths.size(1);
  auto D2 = lengths.size(2);
  auto L = values.numel();
  auto permuted_lengths = at::empty({D2, D0, D1}, lengths.options());
  auto permuted_values = at::empty(L, values.options());
  if (L == 0) {
    return {permuted_lengths, permuted_values};
  }
  AT_DISPATCH_INTEGRAL_TYPES(
      lengths.scalar_type(), "permute_4d_jagged_2013_lengths_cpu_kernel_", [&] {
        using index_t = scalar_t;
        _permute_4d_jagged_2013_lengths_cpu_kernel<index_t>(
            D0,
            D1,
            D2,
            lengths.accessor<index_t, 3>(),
            permuted_lengths.accessor<index_t, 3>());
      });
  const auto offsets =
      fbgemm_gpu::asynchronous_exclusive_cumsum_cpu(lengths.view({-1}))
          .view({D0, D1, D2});
  const auto permuted_offsets = fbgemm_gpu::asynchronous_exclusive_cumsum_cpu(
                                    permuted_lengths.view({-1}))
                                    .view({D2, D0, D1});
  AT_DISPATCH_INTEGRAL_TYPES(
      lengths.scalar_type(),
      "permute_4d_jagged_2013_values_cpu_kernel_input1",
      [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            values.scalar_type(),
            "permute_4d_jagged_2013_values_cpu_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _permute_4d_jagged_2013_values_cpu_kernel<index_t, val_t>(
                  D0,
                  D1,
                  D2,
                  lengths.accessor<index_t, 3>(),
                  offsets.accessor<index_t, 3>(),
                  permuted_offsets.accessor<index_t, 3>(),
                  values.accessor<val_t, 1>(),
                  permuted_values.accessor<val_t, 1>());
            });
      });
  return {permuted_lengths, permuted_values};
}

std::tuple<at::Tensor, at::Tensor> permute_4d_jagged_2013_meta(
    const at::Tensor& lengths,
    const at::Tensor& values) {
  auto D0 = lengths.size(0);
  auto D1 = lengths.size(1);
  auto D2 = lengths.size(2);
  auto L = values.numel();
  return {
      at::native::empty_meta_symint(
          {D2, D0, D1},
          /*dtype=*/::std::make_optional(lengths.scalar_type()),
          /*layout=*/::std::make_optional(lengths.layout()),
          /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
          /*pin_memory=*/::std::nullopt),
      at::native::empty_meta_symint(
          {L},
          /*dtype=*/::std::make_optional(values.scalar_type()),
          /*layout=*/::std::make_optional(values.layout()),
          /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
          /*pin_memory=*/::std::nullopt)};
}
} // namespace gr
