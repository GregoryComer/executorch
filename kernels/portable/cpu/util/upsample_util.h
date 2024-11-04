/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

bool check_upsample_nearest2d_args(
    const Tensor& in, 
    const exec_aten::OptionalArrayRef<int64_t> output_size, 
    const exec_aten::OptionalArrayRef<float> scale_factors, 
    Tensor& out);

}
}