#pragma once

#include <ATen/ATen.h>
#include <optional>

namespace gr {

template <typename key_t, typename val_t>
std::tuple<at::Tensor, at::Tensor> sort_kv_pairs_cuda_dispatched(
    const at::Tensor& keys_contig,
    const at::Tensor& values_contig,
    const std::optional<int64_t>& end_bit,
    const bool descending);

}
