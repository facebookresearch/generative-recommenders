#include "early_stopping.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <string>

namespace mlperf {

namespace loadgen {

double lbeta(int64_t x, int64_t y) {
  return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

// The Gaussian Hypergeometric function specialized for a = 1.
// Based on http://dlmf.nist.gov/15.2.E1.
// Converges if c > 0 and (b <= 0 or x < 1).
// TODO(ckstanton): http://dlmf.nist.gov/15.2.E1 says there are transformations
// to replace x with with a value less than 0.5, for faster convergence.
// Presently, this function can take up to 200,000 iterations to converge.
double hypergeometric_2F1_A1(int64_t b, int64_t c, double x) {
  // TODO(ckstanton): Is there a more principled way to pick kTolerance?
  constexpr double kTolerance = 1.0 / (1LL << 33);
  double term = 1.0;
  double result = 1.0;
  for (int64_t i = 0; std::abs(term) > kTolerance; ++i) {
    term *= (b + i) * x / (c + i);
    result += term;
  }
  return result;
}

// BetaRegularized[x, a, b] =
// Beta[x, a, b]/Beta[a, b] =
// x^a/a Hypergeometric2F1[a, 1-b, 1+a, x]/Beta[a, b] =
// (http://dlmf.nist.gov/15.8.E1.)
// x^a/a (1-x)^(b-1) Hypergeometric2F1[1, 1-b, 1+a, x/(x-1)]/Beta[a, b]
double beta_regularized(double x, int64_t a, int64_t b) {
  return std::exp(a * std::log(x) + (b - 1) * std::log(1 - x) - lbeta(a, b)) /
         a * hypergeometric_2F1_A1(1 - b, 1 + a, x / (x - 1));
}

// Compute the odds of t or fewer overlatency queries in h + t total queries.
// The binomial distribution is the discrete probability distribution for
// independent boolean experiments. The CDF of the binomial distribution is:
// BetaRegularized[q, n - k, 1 + k] where 1 - q is the probability of an event
// per experiment, n is the total number of experiments, and k is the number of
// events. An even in our case is an overlatency query, so q = p - d, n = h + t,
// and k = t.
// Sum[Binomial[h + t, x] (p - d)^(h + t - x) (1 - p + d)^x, {x, 0, t}] =
// BetaRegularized[p - d, h, 1 + t]
double odds(int64_t h, int64_t t, double p, double d) {
  return beta_regularized(p - d, h, 1 + t);
}

// Binary search to find the minimum value h such that:
// odds(h, t, p, d) <= 1 - c on the range [min_h, max_h] given t, p, d, and c.
int64_t find_min_passing(int64_t min_h, int64_t max_h, int64_t t, double p,
                         double d, double c) {
  int64_t count = max_h - min_h;
  while (count > 0) {
    int64_t step = count / 2;
    int64_t h = min_h + step;
    double prob = odds(h, t, p, d);
    if (prob < 1 - c) {
      count = step;
    } else {
      min_h = h + 1;
      count -= step + 1;
    }
  }
  return min_h;
}

int64_t MinPassingQueriesFinder::operator()(int64_t t, double p, double d,
                                            double c) {
  // Given t, p, d, and c, return the minimum h such that odds(h, t, p, d) <= 1
  // - c

  auto &cache = caches_[std::make_tuple(p, d, c)];
  auto it = cache.lower_bound(t);
  if (it != cache.end() && it->first == t) {
    return it->second;
  }

  int64_t x0 = -1;
  int64_t y0 = 0;
  int64_t x1 = 0;
  int64_t y1 = std::ceil(std::log(1 - c) / std::log(p - d));

  if (it != cache.begin()) {
    --it;
    x1 = it->first;
    y1 = it->second;
  }

  if (it != cache.begin()) {
    --it;
    x0 = it->first;
    y0 = it->second;
  }

  double min_slope = (p - d) / (1 - p + d);
  double max_slope = (y1 - y0) * (x1 - x0);
  int64_t min_h = (t - x1) * min_slope + y1;
  int64_t max_h = (t - x1) * max_slope + y1 + 1;
  int64_t h = find_min_passing(min_h, max_h, t, p, d, c);
  cache[t] = h;
  return h;
}

}  // namespace loadgen
}  // namespace mlperf
