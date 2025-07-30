# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import unittest

from transformers import AutoModelForMaskedLM, AutoTokenizer, MobileBertModel

from executorch.backends.test.suite.flow import TestFlow
from executorch.backends.test.suite.models import (
    model_test_cls,
    model_test_params,
    run_model_test,
)


@model_test_cls
class Transformers(unittest.TestCase):
    def test_mobilebert(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = MobileBertModel.from_pretrained(
            "google/mobilebert-uncased", return_dict=False
        )

        inputs = (torch.randint(0, 100, (1, 63)),)

        dynamic_shapes = {
            "input_ids": {
                1: torch.export.Dim("len", min=1, max=128)
            }
        } if use_dynamic_shapes else None

        # Don't use random test inputs as the tokens IDs need to be in range of the vocab size.
        run_model_test(model, inputs, flow, dtype, dynamic_shapes, generate_random_test_inputs=False)
    
    def test_roberta(
        self, flow: TestFlow, dtype: torch.dtype, use_dynamic_shapes: bool
    ):
        model = AutoModelForMaskedLM.from_pretrained(
            "FacebookAI/roberta-base",
            torch_dtype=torch.float,
            attn_implementation="sdpa"
        )

        inputs = (torch.randint(0, 100, (1, 63)),)

        dynamic_shapes = {
            "input_ids": {
                1: torch.export.Dim("len", min=1, max=128)
            }
        } if use_dynamic_shapes else None

        # Don't use random test inputs as the tokens IDs need to be in range of the vocab size.
        run_model_test(model, inputs, flow, dtype, dynamic_shapes, generate_random_test_inputs=False)
