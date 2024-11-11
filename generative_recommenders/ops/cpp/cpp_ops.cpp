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

namespace gr {
at::Tensor expand_1d_jagged_to_dense_cpu(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const int64_t max_len);

at::Tensor expand_1d_jagged_to_dense_meta(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const c10::SymInt max_len);

at::Tensor expand_1d_jagged_to_dense_cuda(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const int64_t max_len);

at::Tensor batched_complete_cumsum_cpu(const at::Tensor& values);

at::Tensor batched_complete_cumsum_cuda(const at::Tensor& values);

at::Tensor batched_complete_cumsum_meta(const at::Tensor& values);

at::Tensor concat_1d_jagged_jagged_cpu(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);

at::Tensor concat_1d_jagged_jagged_cuda(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);

at::Tensor concat_1d_jagged_jagged_meta(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);
} // namespace gr

TORCH_LIBRARY_FRAGMENT(gr, m) {
  m.def(
      "expand_1d_jagged_to_dense(Tensor values, Tensor offsets, SymInt max_len) -> Tensor");
  m.def("batched_complete_cumsum(Tensor values) -> Tensor");
  m.def(
      "concat_1d_jagged_jagged(Tensor lengths_left, Tensor values_left, Tensor lengths_right, Tensor values_right) -> Tensor");
}

TORCH_LIBRARY_IMPL(gr, CPU, m) {
  m.impl("expand_1d_jagged_to_dense", gr::expand_1d_jagged_to_dense_cpu);
  m.impl("batched_complete_cumsum", gr::batched_complete_cumsum_cpu);
  m.impl("concat_1d_jagged_jagged", gr::concat_1d_jagged_jagged_cpu);
}

TORCH_LIBRARY_IMPL(gr, CUDA, m) {
  m.impl("expand_1d_jagged_to_dense", gr::expand_1d_jagged_to_dense_cuda);
  m.impl("batched_complete_cumsum", gr::batched_complete_cumsum_cuda);
  m.impl("concat_1d_jagged_jagged", gr::concat_1d_jagged_jagged_cuda);
}

TORCH_LIBRARY_IMPL(gr, Meta, m) {
  m.impl("expand_1d_jagged_to_dense", gr::expand_1d_jagged_to_dense_meta);
  m.impl("batched_complete_cumsum", gr::batched_complete_cumsum_meta);
  m.impl("concat_1d_jagged_jagged", gr::concat_1d_jagged_jagged_meta);
}

TORCH_LIBRARY_IMPL(gr, Autograd, m) {
  m.impl(
      "expand_1d_jagged_to_dense",
      torch::autograd::autogradNotImplementedFallback());
  m.impl(
      "batched_complete_cumsum",
      torch::autograd::autogradNotImplementedFallback());
}
