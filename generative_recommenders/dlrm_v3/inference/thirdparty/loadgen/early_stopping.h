#ifndef MLPERF_LOADGEN_EARLYSTOPPING_H_
#define MLPERF_LOADGEN_EARLYSTOPPING_H_

#include <cstdint>
#include <map>

namespace mlperf {
namespace loadgen {

class MinPassingQueriesFinder {
 public:
  int64_t operator()(int64_t t, double p, double d, double c);

 private:
  // Memoize prior computations results and use them to bound the binary search
  // range for subsequent computations.

  // TODO: Is there something more efficient to use besides std::map for
  // caches_?
  std::map<std::tuple<double, double, double>, std::map<int64_t, int64_t>>
      caches_;
};

}  // namespace loadgen
}  // namespace mlperf

#endif  // MLPERF_LOADGEN_EARLYSTOPPING_H_
