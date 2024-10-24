#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND4(                \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND4(                                 \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND4(                              \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__))

inline __attribute__((always_inline)) uint32_t
div_round_up(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
};

inline __attribute__((always_inline)) uint32_t next_power_of_2(uint32_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}
