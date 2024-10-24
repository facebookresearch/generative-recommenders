#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace hammer {
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
} // namespace hammer

TORCH_LIBRARY_FRAGMENT(hammer, m) {
  m.def(
      "expand_1d_jagged_to_dense(Tensor values, Tensor offsets, SymInt max_len) -> Tensor");
  m.def("batched_complete_cumsum(Tensor values) -> Tensor");
}

TORCH_LIBRARY_IMPL(hammer, CPU, m) {
  m.impl("expand_1d_jagged_to_dense", hammer::expand_1d_jagged_to_dense_cpu);
  m.impl("batched_complete_cumsum", hammer::batched_complete_cumsum_cpu);
}

TORCH_LIBRARY_IMPL(hammer, CUDA, m) {
  m.impl("expand_1d_jagged_to_dense", hammer::expand_1d_jagged_to_dense_cuda);
  m.impl("batched_complete_cumsum", hammer::batched_complete_cumsum_cuda);
}

TORCH_LIBRARY_IMPL(hammer, Meta, m) {
  m.impl("expand_1d_jagged_to_dense", hammer::expand_1d_jagged_to_dense_meta);
  m.impl("batched_complete_cumsum", hammer::batched_complete_cumsum_meta);
}

TORCH_LIBRARY_IMPL(hammer, Autograd, m) {
  m.impl(
      "expand_1d_jagged_to_dense",
      torch::autograd::autogradNotImplementedFallback());
  m.impl(
      "batched_complete_cumsum",
      torch::autograd::autogradNotImplementedFallback());
}
