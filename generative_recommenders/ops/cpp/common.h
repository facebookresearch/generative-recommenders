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

/*
 * Because different .SO may include the same CUDA CUB kernels, this results in
 * confusion, where libA may end up calling libB's cub kernel and causing
 * failures when we static link libcudart_static.a. To avoid this, we annotate
 * only the public functions and hide the rest.
 */
#define DLL_PUBLIC __attribute__((visibility("default")))
