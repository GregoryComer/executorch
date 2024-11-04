/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <executorch/kernels/test/TestUtil.h>

#include <gtest/gtest.h>

class OpUpsampleNearest2dTest : public OperatorTest {
 protected:
    Tensor& op_upsample_nearest2d_out(
        const Tensor& in, 
        const OptionalArrayRef<int64_t> output_size, 
        const OptionalArrayRef<float> scale_factors, 
        Tensor& out) {
        return torch::executor::aten::upsample_nearest2d_out(context_, input, dim, out);
    }
};

TEST_F(OpUpsampleNearest2dTest, SmokeTest) {
    TensorFactory<ScalarType::Float> tf;

    const auto input = tf.make({1, 1, 2, 2}, {
        0.1, 0.2,
        1.1, 1.2,
    });
    const std::array<int64_t, 2> output_size = { 4, 4 };
    auto output = tf.zeros({1, 1, 4, 4});

    op_upsample_nearest2d_out(input, output_size, {}, out);

    const auto expected = tf.make({1, 1, 4, 4}, {
        0.1, 0.1, 0.2, 0.2,
        0.1, 0.1, 0.2, 0.2,
        1.1, 1.1, 1.2, 1.2,
        1.1, 1.1, 1.2, 1.2,
    });

    EXPECT_TENSOR_EQ(out, expected);
}