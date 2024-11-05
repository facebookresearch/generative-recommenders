# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-unsafe

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    from hammer.ops.triton.utils import (
        _switch_to_contiguous_if_needed,
        register_tritoncc_specs,
        triton_autotune,
        VersionedSpec,
    )
except ImportError:
    from hammer.oss.generative_recommenders.ops.triton.utils import (
        _switch_to_contiguous_if_needed,
        register_tritoncc_specs,
        triton_autotune,
        VersionedSpec,
    )

from hammer.oss.generative_recommenders.ops.triton.triton_addmm import (
    _addmm_fwd,
    get_mm_configs,
)


def _get_ln_mul_dropout_named_specs() -> List[VersionedSpec]:
    s: int = 16
    dtype: str = "*bf16"
    TRAINING: bool = False
    return (
        [
            VersionedSpec(
                spec={
                    "X": (dtype, s),
                    "U": (dtype, s),
                    "Y": (dtype, s),
                    "W": (dtype, s),
                    "B": (dtype, s),
                    "Mean": (dtype, s),
                    "Rstd": (dtype, s),
                    "D": ("i32", s),
                    "eps": "fp32",
                    "seed": "i64",
                    "dropout_ratio": "fp32",
                    "stride_x": ("i32", s),
                    "stride_u": ("i32", s),
                    "stride_y": ("i32", s),
                    "BLOCK_D": BLOCK_D,
                    "TRAINING": TRAINING,
                    "CONCAT_UX": CONCAT_UX,
                }
            )
            for BLOCK_D in [256, 512]
            for CONCAT_UX in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "X": (dtype, s),
                    "U": (dtype, s),
                    "Y": (dtype, s),
                    "W": (dtype, s),
                    "B": (dtype, s),
                    "Mean": (dtype, s),
                    "Rstd": (dtype, s),
                    "D": ("i32", s),
                    "eps": "fp32",
                    "seed": "i64",
                    "dropout_ratio": "fp32",
                    "stride_x": ("i32", s),
                    "stride_u": ("i32", s),
                    "stride_y": ("i32", s),
                    "BLOCK_D": BLOCK_D,
                    "TRAINING": TRAINING,
                    "CONCAT_UX": CONCAT_UX,
                },
                version="standalone_cint_v1",
            )
            for BLOCK_D in [256, 512]
            for CONCAT_UX in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "X": (dtype, s),
                    "U": (dtype, s),
                    "Y": (dtype, s),
                    "W": (dtype, s),
                    "B": (dtype, s),
                    "Mean": (dtype, s, False),
                    "Rstd": (dtype, s, False),
                    "D": ("i32", s),
                    "eps": "fp32",
                    "seed": "i64",
                    "dropout_ratio": "fp32",
                    "stride_x": ("i32", s),
                    "stride_u": ("i32", s),
                    "stride_y": ("i32", s),
                    "BLOCK_D": BLOCK_D,
                    "TRAINING": TRAINING,
                    "CONCAT_UX": CONCAT_UX,
                },
                version="standalone_cint_v2",
            )
            for BLOCK_D in [256, 512]
            for CONCAT_UX in [True, False]
        ]
    )


def _get_group_norm_mul_dropout_named_specs() -> List[VersionedSpec]:
    s: int = 16
    dtype: str = "*bf16"
    TRAINING: bool = False
    return (
        [
            VersionedSpec(
                spec={  # pyre-ignore [6]
                    "X": (dtype, s),
                    "U": (dtype, s),
                    "Y": (dtype, s),
                    "W": (dtype, s),
                    "B": (dtype, s),
                    "Mean": (dtype, s),
                    "Rstd": (dtype, s),
                    "D": ("i32", s),
                    "Heads": 4,
                    "eps": "fp32",
                    "seed": "i64",
                    "dropout_ratio": "fp32",
                    "stride_x": ("i32", s),
                    "stride_u": ("i32", s),
                    "stride_y": ("i32", s),
                    "BLOCK_D": BLOCK_D,
                    "BLOCK_H": 4,
                    "TRAINING": TRAINING,
                    "CONCAT_UX": CONCAT_UX,
                }
            )
            for BLOCK_D in [128, 256, 512]
            for CONCAT_UX in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "X": (dtype, s),
                    "U": (dtype, s),
                    "Y": (dtype, s),
                    "W": (dtype, s),
                    "B": (dtype, s),
                    "Mean": (dtype, s),
                    "Rstd": (dtype, s),
                    "D": ("i32", s),
                    "Heads": 4,
                    "eps": "fp32",
                    "seed": "i64",
                    "dropout_ratio": "fp32",
                    "stride_x": ("i32", s),
                    "stride_u": ("i32", s),
                    "stride_y": ("i32", s),
                    "BLOCK_D": BLOCK_D,
                    "BLOCK_H": 4,
                    "TRAINING": TRAINING,
                    "CONCAT_UX": CONCAT_UX,
                },
                version="standalone_cint_v1_gn",
            )
            for BLOCK_D in [128, 256, 512]
            for CONCAT_UX in [True, False]
        ]
        + [
            VersionedSpec(
                spec={  # pyre-ignore [6]
                    "X": (dtype, s),
                    "U": (dtype, s),
                    "Y": (dtype, s),
                    "W": (dtype, s),
                    "B": (dtype, s),
                    "Mean": (dtype, s, False),
                    "Rstd": (dtype, s, False),
                    "D": ("i32", s),
                    "Heads": 4,
                    "eps": "fp32",
                    "seed": "i64",
                    "dropout_ratio": "fp32",
                    "stride_x": ("i32", s),
                    "stride_u": ("i32", s),
                    "stride_y": ("i32", s),
                    "BLOCK_D": BLOCK_D,
                    "BLOCK_H": 4,
                    "TRAINING": TRAINING,
                    "CONCAT_UX": CONCAT_UX,
                },
                version="standalone_cint_v2",
            )
            for BLOCK_D in [128, 256, 512]
            for CONCAT_UX in [True, False]
        ]
    )


def _get_addmm_named_specs() -> List[VersionedSpec]:
    s: int = 16
    dtype: str = "*bf16"
    default_values = {
        "BROADCAST_Y": 0,
    }
    return (
        [
            VersionedSpec(
                spec={
                    "x_ptr": (dtype, s),
                    "w_ptr": (dtype, s),
                    "y_ptr": (dtype, s),
                    "z_ptr": (dtype, s),
                    "M": "i32",
                    "N": ("i32", s),
                    "K": ("i32", s),
                    "stride_xm": ("i32", s),
                    "stride_xk": ("i32", 1),
                    "stride_wk": ("i32", s),
                    "stride_wn": ("i32", 1),
                    "stride_ym": ("i32", s),
                    "stride_yn": ("i32", 1),
                    "stride_zm": ("i32", s),
                    "stride_zn": ("i32", 1),
                    "BLOCK_M": -1,  # autotuned
                    "BLOCK_N": -1,  # autotuned
                    "BLOCK_K": -1,  # autotuned
                    "GROUP_M": -1,  # autotuned
                    "ALLOW_TF32": True,
                    "BROADCAST_Y": False,
                },
                default_values=default_values,
                version="standalone_cint_v1",
            )
        ]
        + [
            VersionedSpec(
                spec={
                    "x_ptr": (dtype, s),
                    "w_ptr": (dtype, s),
                    "y_ptr": (dtype, s),
                    "z_ptr": (dtype, s),
                    "M": "i32",
                    "N": ("i32", s),
                    "K": ("i32", s),
                    "stride_xm": ("i32", s),
                    "stride_xk": ("i32", 1),
                    "stride_wk": ("i32", s),
                    "stride_wn": ("i32", 1),
                    "stride_ym": ("i32", s),
                    "stride_yn": ("i32", 1),
                    "stride_zm": ("i32", s),
                    "stride_zn": ("i32", 1),
                    "BLOCK_M": -1,  # autotuned
                    "BLOCK_N": -1,  # autotuned
                    "BLOCK_K": -1,  # autotuned
                    "GROUP_M": -1,  # autotuned
                    "ALLOW_TF32": True,
                    "BROADCAST_Y": broadcast_y,
                },
                default_values=default_values,
            )
            for broadcast_y in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "x_ptr": (dtype, s),
                    "w_ptr": (dtype, s),
                    "y_ptr": (dtype, s),
                    "z_ptr": (dtype, s),
                    "M": "i32",
                    "N": ("i32", s),
                    "K": ("i32", s),
                    "stride_xm": ("i32", s),
                    "stride_xk": ("i32", 1),
                    "stride_wk": ("i32", s),
                    "stride_wn": ("i32", 1),
                    "stride_ym": ("i32", s),
                    "stride_yn": ("i32", 1),
                    "stride_zm": ("i32", s),
                    "stride_zn": ("i32", 1),
                    "BLOCK_M": -1,  # autotuned
                    "BLOCK_N": -1,  # autotuned
                    "BLOCK_K": -1,  # autotuned
                    "GROUP_M": -1,  # autotuned
                    "ALLOW_TF32": True,
                    "BROADCAST_Y": broadcast_y,
                },
                version="standalone_cint_v2",
                default_values=default_values,
            )
            for broadcast_y in [True, False]
        ]
    )


@triton.jit
def _ln_mul_dropout_fwd(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)

    # Compute mean
    mean = 0.0
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D

    # Compute variance
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = y * w + b
    u = tl.load(U + cols, mask=cols < D, other=0.0).to(tl.float32)
    y = y * u

    if TRAINING:
        random_offsets = row * BLOCK_D + cols
        if CONCAT_UX:
            # apply dropout on u
            random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on x
            random_x = tl.rand(seed, random_offsets + D)
            x_keep = random_x > dropout_ratio
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            random_y = tl.rand(seed, random_offsets + 2 * D)
            y_keep = random_y > dropout_ratio
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random = tl.rand(seed, random_offsets)
            y_keep = random > dropout_ratio
            # write-back
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    # Write output
    if CONCAT_UX:
        tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)

_ln_mul_dropout_fwd = register_tritoncc_specs(
    func=_ln_mul_dropout_fwd, versioned_specs=_get_ln_mul_dropout_named_specs()
)

@triton.jit
def _ln_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D,
    eps,
    seed,
    dropout_ratio,
    N,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    if rows_per_tile == 0:
        return

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols

    for idx in range(0, rows_per_tile):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_UX:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if TRAINING:
            random_offsets = row * BLOCK_D + cols
            if CONCAT_UX:
                # apply dropout on du
                random_du = tl.rand(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dx
                random_dx = tl.rand(seed, random_offsets + D)
                dx_keep = random_dx > dropout_ratio
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dy
                random_dy = tl.rand(seed, random_offsets + 2 * D)
                dy_keep = random_dy > dropout_ratio
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            else:
                random = tl.rand(seed, random_offsets)
                dy_keep = random > dropout_ratio
                # write-back
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)

        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        b = tl.load(B + cols, mask=mask).to(tl.float32)
        ln = xhat * w + b
        du += dy * ln
        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        dy = dy * u
        wdy = w * dy
        if COMPUTE_Y:
            y = ln * u
            if TRAINING:
                if CONCAT_UX:
                    u = tl.where(du_keep, u / (1.0 - dropout_ratio), 0.0)  # pyre-ignore [61]
                    x = tl.where(dx_keep, x / (1.0 - dropout_ratio), 0.0)  # pyre-ignore [61]
                    y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)  # pyre-ignore [61]
                else:
                    y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)  # pyre-ignore [61]
            if CONCAT_UX:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y

        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / D
        c2 = tl.sum(wdy, axis=0) / D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw = dy * xhat
        partial_db = dy
        # First store doesn't accumulate
        if idx > 0:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)
        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du
        row += tile_num


def _get_bwd_dwdb_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [32, 64, 128, 256]:
        for num_warps in [8, 16] + ([] if torch.ops.hip else [32]):
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=["D"],
)
@triton.jit
def _ln_mul_dropout_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)


def triton_layer_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    concat_ux: bool = False,
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int
]:  # y, mean, rstd, BLOCK_D, num_warps, seed
    assert x.dim() == 2
    x = _switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == D
    assert bias.numel() == D

    if concat_ux:
        y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, 0, 0, 0
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if seed is None:
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    # pyre-ignore[28]
    _ln_mul_dropout_fwd[(N,)](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        D,
        eps,
        seed,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        BLOCK_D=BLOCK_D,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        num_warps=num_warps,
    )
    return y, mean, rstd, BLOCK_D, num_warps, seed  # pyre-ignore [7]


def triton_layer_norm_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: Optional[int] = None,
    concat_ux: bool = False,
    compute_y: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    y = None
    N, D = x.shape
    if compute_y:
        if concat_ux:
            y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
        else:
            y = torch.empty_like(x)
    if N == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(u),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            y,
        )
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num = max(1, min(sms * 64, N // 4))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    # pyre-ignore[28]
    _ln_mul_dropout_bwd_dx_du[(tile_num,)](
        dx,
        du,
        dy,
        _dweight,
        _dbias,
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        du.stride(0),
        dy.stride(0),
        x.stride(0),
        u.stride(0),
        y.stride(0) if compute_y else 0,  # pyre-ignore [16]
        D,
        eps,
        seed,
        dropout_ratio,
        N=N,
        BLOCK_D=BLOCK_D,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        COMPUTE_Y=compute_y,
        num_warps=num_warps,
    )

    def grid(META):
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
    BLOCK_D = min(max(BLOCK_D, 4), 128)
    _ln_mul_dropout_bwd_dwdb[grid](
        _dweight,
        _dbias,
        dweight,
        dbias,
        tile_num,
        D,
        BLOCK_D=BLOCK_D,
    )
    return dx, du, dweight, dbias, y


class LayerNormMulDropoutFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        concat_ux: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        y, mean, rstd, BLOCK_D, num_warps, seed = triton_layer_norm_mul_dropout_fwd(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=eps,
            dropout_ratio=dropout_ratio,
            training=training,
            concat_ux=concat_ux,
            seed=seed,
        )
        ctx.save_for_backward(x, u, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dy: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, u, weight, bias, mean, rstd = ctx.saved_tensors
        dx, du, dweight, dbias, _ = triton_layer_norm_mul_dropout_bwd(
            dy=dy,
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=ctx.num_warps,
            eps=ctx.eps,
            training=ctx.training,
            dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed,
            concat_ux=ctx.concat_ux,
            compute_y=False,
        )
        return dx, du, dweight, dbias, None, None, None, None, None


@triton.jit
def _group_norm_mul_dropout_fwd(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    Heads,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_H)
    offsets = heads[:, None] * D + cols[None, :]
    mask_h = heads < Heads
    mask_c = cols < D
    mask = mask_c[None, :] & mask_h[:, None]

    # Compute mean
    mean = 0.0
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    mean = tl.ravel(mean)

    # Compute variance
    _var = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(mask, x - mean[:, None], 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=1) / D
    var = tl.ravel(var)
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row * Heads + heads, mean, mask=mask_h)
    tl.store(Rstd + row * Heads + heads, rstd, mask=mask_h)

    # Normalize and apply linear transformation
    y = x_mean * rstd[:, None]  # pyre-ignore [16]
    w = tl.load(W + heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + heads, mask=mask_h).to(tl.float32)
    y = y * w[:, None] + b[:, None]
    u = tl.load(U + offsets, mask=mask, other=0.0).to(tl.float32)
    y = y * u

    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            # apply dropout on u
            random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on x
            random_x = tl.rand(seed, random_offsets + Heads * D)
            x_keep = random_x > dropout_ratio
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            random_y = tl.rand(seed, random_offsets + 2 * Heads * D)
            y_keep = random_y > dropout_ratio
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            y_keep = random > dropout_ratio
            # write-back
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    # Write output
    if CONCAT_UX:
        tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


_group_norm_mul_dropout_fwd = register_tritoncc_specs(
    func=_group_norm_mul_dropout_fwd, versioned_specs=_get_group_norm_mul_dropout_named_specs()
)

@triton.jit
def _group_norm_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D,
    Heads,
    eps,
    seed,
    dropout_ratio,
    GROUP_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    off_heads = tl.arange(0, BLOCK_H)
    mask_c = cols < D
    mask_h = off_heads < Heads
    mask = mask_c[None, :] & mask_h[:, None]
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    offsets = off_heads[:, None] * D + cols[None, :]

    # Load data to SRAM
    x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
    if CONCAT_UX:
        du = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
        dx = tl.load(DY + Heads * D + offsets, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + 2 * Heads * D + offsets, mask=mask, other=0).to(tl.float32)
    else:
        du = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dx = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dy = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            # apply dropout on du
            random_du = tl.rand(seed, random_offsets)
            du_keep = random_du > dropout_ratio
            du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
            # apply dropout on dx
            random_dx = tl.rand(seed, random_offsets + Heads * D)
            dx_keep = random_dx > dropout_ratio
            dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
            # apply dropout on dy
            random_dy = tl.rand(seed, random_offsets + 2 * Heads * D)
            dy_keep = random_dy > dropout_ratio
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            dy_keep = random > dropout_ratio
            # write-back
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)

    mean = tl.load(Mean + row * Heads + off_heads)
    rstd = tl.load(Rstd + row * Heads + off_heads)

    # Compute dx
    xhat = (x - mean[:, None]) * rstd[:, None]
    w = tl.load(W + off_heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + off_heads, mask=mask_h).to(tl.float32)
    ln = xhat * w[:, None] + b[:, None]
    du += dy * ln
    tl.store(DU + offsets, du.to(DU.dtype.element_ty), mask=mask)
    u = tl.load(U + offsets, mask=mask, other=0).to(tl.float32)
    dy = dy * u
    wdy = w[:, None] * dy
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
        y = ln * u
        if TRAINING:
            if CONCAT_UX:
                u = tl.where(du_keep, u / (1.0 - dropout_ratio), 0.0)  # pyre-ignore [61]
                x = tl.where(dx_keep, x / (1.0 - dropout_ratio), 0.0)  # pyre-ignore [61]
                y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)  # pyre-ignore [61]
            else:
                y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)  # pyre-ignore [61]
        if CONCAT_UX:
            tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty), mask=mask)
        else:
            tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)

    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=1) / D
    c2 = tl.sum(wdy, axis=1) / D
    dx += (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    # Write dx
    tl.store(DX + offsets, dx, mask=mask)

    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_N
    DW = DW + lock_id * Heads + off_heads
    DB = DB + lock_id * Heads + off_heads
    # Accumulate partial sums for dw/db
    partial_dw = tl.sum(dy * xhat, axis=1)
    partial_dw = tl.ravel(partial_dw)
    partial_db = tl.sum(dy, axis=1)
    partial_db = tl.ravel(partial_db)
    tl.atomic_add(
        DW,
        partial_dw,
        mask=mask_h,
        sem="relaxed",
    )
    tl.atomic_add(
        DB,
        partial_db,
        mask=mask_h,
        sem="relaxed",
    )


def triton_group_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    concat_ux: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int
]:  # y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed
    assert x.dim() == 2
    assert x.shape == u.shape
    assert x.shape[1] == num_heads * linear_dim
    x = _switch_to_contiguous_if_needed(x)
    u = _switch_to_contiguous_if_needed(u)
    N, _ = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == num_heads
    assert bias.numel() == num_heads

    if concat_ux:
        y = torch.empty((N, 3 * num_heads * linear_dim), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype, device=x.device)
    mean = torch.empty((N * num_heads,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N * num_heads,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, 0, 0, 0, 0
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = triton.next_power_of_2(linear_dim)
    BLOCK_H: int = triton.next_power_of_2(num_heads)
    if BLOCK_D * BLOCK_H > MAX_FUSED_SIZE:
        raise RuntimeError(
            "This group norm doesn't support num_heads * linear_dim >= 64KB."
        )

    if seed is None:
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()
    num_warps: int = min(max(BLOCK_D * BLOCK_H // 256, 1), 8)
    # pyre-ignore[28]
    _group_norm_mul_dropout_fwd[(N,)](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        linear_dim,
        num_heads,
        eps,
        seed,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        num_warps=num_warps,
    )
    return y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed  # pyre-ignore [7]


def triton_group_norm_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    BLOCK_H: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: Optional[int] = None,
    concat_ux: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    compute_y: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    y = None
    N, dim = x.shape
    if compute_y:
        if concat_ux:
            y = torch.empty(
                (N, 3 * num_heads * linear_dim), dtype=x.dtype, device=x.device
            )
        else:
            y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype, device=x.device)
    if N == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(u),
            torch.zeros_like(weight),
            torch.zeros_like(bias),
            y,
        )
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    if dim <= 1024:
        GROUP_N = 256 * 8
    elif dim <= 4096:
        GROUP_N = 128 * 8
    elif dim <= 8192:
        GROUP_N = 96 * 8
    else:
        GROUP_N = 64 * 8
    GROUP_N = N if GROUP_N > N else GROUP_N
    _dweight = torch.zeros((GROUP_N, num_heads), dtype=torch.float32, device=x.device)
    _dbias = torch.zeros((GROUP_N, num_heads), dtype=torch.float32, device=x.device)
    dweight = torch.empty((num_heads,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((num_heads,), dtype=weight.dtype, device=x.device)
    # pyre-ignore[28]
    _group_norm_mul_dropout_bwd_dx_du[(N,)](
        dx,
        du,
        dy,
        _dweight,
        _dbias,
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        du.stride(0),
        dy.stride(0),
        x.stride(0),
        u.stride(0),
        y.stride(0) if compute_y else 0,  # pyre-ignore [16]
        linear_dim,
        num_heads,
        eps,
        seed,
        dropout_ratio,
        GROUP_N=GROUP_N,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        COMPUTE_Y=compute_y,
        num_warps=num_warps,
    )
    _group_norm_bwd_dwdb[(num_heads,)](
        _dweight,
        _dbias,
        dweight,
        dbias,
        GROUP_N,
    )
    return dx, du, dweight, dbias, y


def _get_bwd_dwdb_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [32, 64, 128, 256]:
        for num_warps in [8, 16] + ([] if torch.ops.hip else [32]):
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs

@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=[],
)
@triton.jit
def _group_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    BLOCK_N: tl.constexpr,
):
    col = tl.program_id(0)
    num_heads = tl.num_programs(0)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = rows < N
        offs = rows * num_heads + col
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + col, sum_dw.to(FINAL_DW.dtype.element_ty))
    tl.store(FINAL_DB + col, sum_db.to(FINAL_DB.dtype.element_ty))

class GroupNormMulDropoutFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        concat_ux: bool = False,
        num_heads: int = 1,
        linear_dim: int = -1,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
            triton_group_norm_mul_dropout_fwd(
                x=x,
                u=u,
                weight=weight,
                bias=bias,
                eps=eps,
                dropout_ratio=dropout_ratio,
                training=training,
                concat_ux=concat_ux,
                num_heads=num_heads,
                linear_dim=linear_dim,
                seed=seed,
            )
        )
        ctx.save_for_backward(x, u, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.BLOCK_H = BLOCK_H
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim = linear_dim
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dy: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, u, weight, bias, mean, rstd = ctx.saved_tensors
        dx, du, dweight, dbias, _ = triton_group_norm_mul_dropout_bwd(
            dy=dy,
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            BLOCK_D=ctx.BLOCK_D,
            BLOCK_H=ctx.BLOCK_H,
            num_warps=ctx.num_warps,
            eps=ctx.eps,
            training=ctx.training,
            dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed,
            concat_ux=ctx.concat_ux,
            num_heads=ctx.num_heads,
            linear_dim=ctx.linear_dim,
            compute_y=False,
        )
        return (
            dx,
            du,
            dweight,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_addmm_fwd = register_tritoncc_specs(
    func=_addmm_fwd, versioned_specs=_get_addmm_named_specs()
)
_addmm_fwd = triton_autotune(
    configs=get_mm_configs(),
    key=[
        "N",
        "K",
    ],
)(_addmm_fwd.fn)



def triton_addmm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    KB, N = w.shape
    assert K == KB, f"incompatible dimensions {K}, {KB}"

    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f"incompatible dimensions {N}, {NY}"

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _addmm_fwd[grid](
        x,
        w,
        y,
        z,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0) if not is_y_1d else 0,
        y.stride(1) if not is_y_1d else y.stride(0),
        z.stride(0),
        z.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
    )
    return z


def triton_addmm_bwd(
    x: torch.Tensor,
    w: torch.Tensor,
    dz: torch.Tensor,
    is_y_1d: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_y_1d:
        dy = torch.sum(dz, dim=0)
    else:
        dy = dz
    dw = torch.mm(x.t(), dz)
    dx = torch.mm(dz, w.t())

    return dx, dw, dy


class _AddMmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.is_y_1d = y.dim() == 1
        return triton_addmm_fwd(x, w, y)

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (x, w) = ctx.saved_tensors
        return triton_addmm_bwd(x, w, dz, ctx.is_y_1d)


class HSTUComputeOutputFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        attn: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        output_weight: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        concat_ux: bool = False,
        group_norm: bool = False,
        num_heads: int = 1,
        linear_dim: int = -1,
        seed: Optional[int] = None,
        recompute_y_in_backward: bool = False,
    ) -> torch.Tensor:
        if group_norm:
            y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
                triton_group_norm_mul_dropout_fwd(
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=eps,
                    dropout_ratio=dropout_ratio,
                    training=training,
                    concat_ux=concat_ux,
                    num_heads=num_heads,
                    linear_dim=linear_dim,
                    seed=seed,
                )
            )
            ctx.BLOCK_H = BLOCK_H
        else:
            y, mean, rstd, BLOCK_D, num_warps, seed = triton_layer_norm_mul_dropout_fwd(
                x=attn,
                u=u,
                weight=norm_weight,
                bias=norm_bias,
                eps=eps,
                dropout_ratio=dropout_ratio,
                training=training,
                concat_ux=concat_ux,
                seed=seed,
            )
        out = triton_addmm_fwd(x=y, w=output_weight, y=x)
        saved_tensors = [attn, u, norm_weight, norm_bias, mean, rstd, output_weight]
        if not recompute_y_in_backward:
            saved_tensors.append(y)
        ctx.save_for_backward(*saved_tensors)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim = linear_dim
        ctx.group_norm = group_norm
        ctx.recompute_y_in_backward = recompute_y_in_backward
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor) -> Tuple[
        torch.Tensor,  # dattn
        torch.Tensor,  # du
        torch.Tensor,  # dx
        torch.Tensor,  # d_norm_weight
        torch.Tensor,  # d_norm_bias
        torch.Tensor,  # d_output_weight
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        attn, u, norm_weight, norm_bias, mean, rstd, output_weight = ctx.saved_tensors[
            :7
        ]
        dy = torch.mm(dout, output_weight.t())

        if ctx.group_norm:
            dattn, du, d_norm_weight, d_norm_bias, y = (
                triton_group_norm_mul_dropout_bwd(
                    dy=dy,
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    mean=mean,
                    rstd=rstd,
                    BLOCK_D=ctx.BLOCK_D,
                    BLOCK_H=ctx.BLOCK_H,
                    num_warps=ctx.num_warps,
                    eps=ctx.eps,
                    training=ctx.training,
                    dropout_ratio=ctx.dropout_ratio,
                    seed=ctx.seed,
                    concat_ux=ctx.concat_ux,
                    num_heads=ctx.num_heads,
                    linear_dim=ctx.linear_dim,
                    compute_y=ctx.recompute_y_in_backward,
                )
            )
        else:
            dattn, du, d_norm_weight, d_norm_bias, y = (
                triton_layer_norm_mul_dropout_bwd(
                    dy=dy,
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    mean=mean,
                    rstd=rstd,
                    BLOCK_D=ctx.BLOCK_D,
                    num_warps=ctx.num_warps,
                    eps=ctx.eps,
                    training=ctx.training,
                    dropout_ratio=ctx.dropout_ratio,
                    seed=ctx.seed,
                    concat_ux=ctx.concat_ux,
                    compute_y=ctx.recompute_y_in_backward,
                )
            )
        if not ctx.recompute_y_in_backward:
            y = ctx.saved_tensors[7]
        d_output_weight = torch.mm(y.t(), dout)
        return (
            dattn,
            du,
            dout,
            d_norm_weight,
            d_norm_bias,
            d_output_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
