// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_stdlib>
#include "gemm/host.h"

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
