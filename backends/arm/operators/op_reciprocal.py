# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import List

import torch

import tosa_tools.v0_80.serializer.tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa_mapping import TosaArg
from executorch.backends.arm.tosa_specification import TosaSpecification


@register_node_visitor
class ReciprocalVisitor_080_MI(NodeVisitor):
    target = "aten.reciprocal.default"

    # BI case should be handled by op_table
    tosa_specs = [TosaSpecification.create_from_string("TOSA-0.80+MI")]

    def __init__(self, *args):
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: ts.TosaSerializer,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        assert inputs[0].dtype == output.dtype == ts.DType.FP32
        tosa_graph.addOperator(
            ts.TosaOp.Op().RECIPROCAL, [inputs[0].name], [output.name]
        )
