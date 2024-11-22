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
#include <ATen/core/op_registration/op_registration.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "fbgemm_gpu/sparse_ops.h" // @manual

namespace gr {

at::Tensor complete_cumsum_cpu(const at::Tensor& values) {
  TORCH_CHECK(values.dim() == 1);
  auto len = values.size(0);
  const torch::Tensor index = at::range(0, len, at::kLong).cpu();
  auto output = fbgemm_gpu::asynchronous_complete_cumsum_cpu(values);
  return output;
}

at::Tensor complete_cumsum_meta(const at::Tensor& values) {
  auto len = values.sym_size(0);
  auto output = at::native::empty_meta_symint(
      {len + 1},
      /*dtype=*/::std::make_optional(values.scalar_type()),
      /*layout=*/::std::make_optional(values.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  return output;
}

} // namespace gr
