# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import unittest

from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.test.suite.models import (
    model_test_cls,
    model_test_params,
    run_model_test,
)

from torch.export import Dim
from torchsr.models import edsr_r16f64


@model_test_cls
class Misc(unittest.TestCase):
    @model_test_params(dtypes=[torch.float32])
    def test_edsr(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = edsr_r16f64(2, True)
        inputs = (torch.randn(1, 3, 224, 224, dtype=dtype),)

        dynamic_shapes = (
            (
                {
                    2: Dim("height", min=1, max=16) * 16,
                    3: Dim("width", min=1, max=16) * 16,
                },
            )
            if use_dynamic_shapes
            else None
        )

        run_model_test(model, inputs, flow, dtype, dynamic_shapes)
    
    # Dynamic sequence length LSTM is not (yet) exportable.
    @model_test_params(supports_dynamic_shapes=False, dtypes=[torch.float32])
    def test_lstm(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(53, 67, 11)

            def forward(self, x):
                return self.lstm(x)
            
        assert not use_dynamic_shapes

        model = LSTMModel()
        inputs = (torch.randn(13, 1, 53),)
        run_model_test(model, inputs, flow, dtype, dynamic_shapes=None)
