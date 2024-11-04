/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/upsample_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace torch {
namespace executor {

bool check_upsample_nearest2d_args(
    const Tensor& in, 
    const exec_aten::OptionalArrayRef<int64_t> output_size, 
    const exec_aten::OptionalArrayRef<float> scale_factors, 
    Tensor& out) {

    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
    ET_LOG_AND_RETURN_IF_FALSE(in.dim() == 4);
    ET_LOG_AND_RETURN_IF_FALSE(out.dim() == 4);
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(in));
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_dim_order(out));
    ET_LOG_AND_RETURN_IF_FALSE(
        (output_size.has_value() && !scale_factors.has_value()) ||
        (!output_size.has_value() && scale_factors.has_value()));

    if (output_size.has_value()) {
        ET_LOG_AND_RETURN_IF_FALSE(output_size.value().size() == 2);
    }

    if (scale_factors.has_value()) {
        ET_LOG_AND_RETURN_IF_FALSE(scale_factors.value().size() > 0 && 
            scale_factors.value().size() <= 2);
    }

    return true;
}

}}