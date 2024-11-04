/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <executorch/kernels/portable/cpu/util/upsample_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& upsample_nearest2d_out(
    KernelRuntimeContext& ctx, 
    const Tensor& in, 
    const exec_aten::OptionalArrayRef<int64_t> output_size, 
    const exec_aten::OptionalArrayRef<float> scale_factors, 
    Tensor& out) {

  ET_KERNEL_CHECK(
      ctx,
      check_upsample_nearest2d_args(in, output_size, scale_factors, out),
      InvalidArgument, 
      out);
    
    // Either output_size or scale_factors are provided, not both. This
    // is checked in check_..._args.
    std::array<Tensor::SizesType, kTensorDimensionLimit> target_size;
    std::array<float, 2> scales;

    const auto dim = in.dim();    
    std::copy(in.sizes().cbegin(), in.sizes().cend(), target_size.begin());

    if (scale_factors.has_value()) {
        std::copy_n(scale_factors.value().cbegin(), 2, scales.begin());

        target_size[dim - 2] = in.sizes()[dim - 2] * scales[0];
        target_size[dim - 1] = in.sizes()[dim - 1] * scales[1];
    }
    else {
        scales[0] = static_cast<float>(output_size.value()[0]) / in.sizes()[dim - 2];
        scales[1] = static_cast<float>(output_size.value()[1]) / in.sizes()[dim - 1];

        target_size[dim - 2] = output_size.value()[0];
        target_size[dim - 1] = output_size.value()[1];
    }

    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(out, {target_size.data(), static_cast<size_t>(dim)}) == Error::Ok,
        InvalidArgument,
        out);

    return out;
}

}
}
}