#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "fbgemm_gpu/sparse_ops.h" // @manual

namespace hammer {

at::Tensor batched_complete_cumsum_cpu(const at::Tensor& values) {
  auto B = values.size(0);
  auto len = values.size(1);
  auto output = at::empty({B, len + 1}, values.options());
  const torch::Tensor index = at::range(0, len, at::kLong).cpu();
  for (auto i : c10::irange(B)) {
    torch::Tensor t = output[i];
    at::index_put_(
        t, {index}, fbgemm_gpu::asynchronous_complete_cumsum_cpu(values[i]));
  }
  return output;
}

at::Tensor batched_complete_cumsum_meta(const at::Tensor& values) {
  auto B = values.sym_size(0);
  auto len = values.sym_size(1);
  auto output = at::native::empty_meta_symint(
      {B, len + 1},
      /*dtype=*/::std::make_optional(values.scalar_type()),
      /*layout=*/::std::make_optional(values.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
  return output;
}

} // namespace hammer
