#include "common.h"
#include "sort_kv_pairs_cuda_kernels_template.h"

namespace gr {

DLL_PUBLIC std::tuple<at::Tensor, at::Tensor> sort_kv_pairs_cuda(
    const at::Tensor& keys,
    const at::Tensor& values,
    const std::optional<int64_t>& end_bit,
    const bool descending = false) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(keys.get_device());
  TORCH_CHECK(
      keys.dtype() == at::kInt || keys.dtype() == at::kLong ||
      keys.dtype() == at::kByte || keys.dtype() == at::kShort);
  TORCH_CHECK(keys.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(keys.dim() == 1);
  TORCH_CHECK(values.dim() == 1);
  at::Tensor sorted_keys;
  at::Tensor sorted_values;

  AT_DISPATCH_INTEGRAL_TYPES(keys.scalar_type(), "sort_pairs_cuda_input1", [&] {
    using key_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        values.scalar_type(),
        "sort_pairs_cuda_input2",
        [&] {
          using val_t = scalar_t;
          std::tie(sorted_keys, sorted_values) =
              sort_kv_pairs_cuda_dispatched<key_t, val_t>(
                  keys, values, end_bit, descending);
        });
  });

  return {std::move(sorted_keys), std::move(sorted_values)};
}

} // namespace gr
