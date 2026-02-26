# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir.lowered_backend_module import (
    arrange_graph_placeholders,
    create_submodule_from_nodes,
)
from torch.export.graph_signature import InputKind


class TestArrangeGraphPlaceholders(unittest.TestCase):
    TAG = "test_tag"

    def setUp(self):
        torch._dynamo.reset()

    def test_constant_tensor_ordered_before_user_inputs(self):
        """arrange_graph_placeholders should place CONSTANT_TENSOR placeholders
        before USER_INPUT placeholders.

        When fuse_as_graphmodule extracts a partition subgraph, external inputs
        become placeholders in argument order. If a user input appears before a
        lifted tensor constant in the op's arguments, the subgraph will have the
        user input placeholder first. arrange_graph_placeholders must fix this
        by sorting the constant into the constant bucket."""

        class ConvScale(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
                self.scale = torch.ones(1, 8, 1, 1)

            def forward(self, x):
                return self.conv(x) * self.scale

        model = ConvScale().eval()
        ep = torch.export.export(model, (torch.randn(1, 3, 4, 4),), strict=True)
        edge_ep = to_edge(ep).exported_program()

        # Tag only the mul op. The conv is not partitioned, so its output
        # becomes a user input to the partition. The constant (self.scale)
        # is left untagged and also becomes an external input.
        for node in edge_ep.graph_module.graph.nodes:
            if node.op == "call_function" and "mul" in str(node.target):
                node.meta["delegation_tag"] = self.TAG

        node_list = [
            n
            for n in edge_ep.graph_module.graph.nodes
            if n.meta.get("delegation_tag", "") == self.TAG
        ]

        # create_submodule_from_nodes calls fuse_as_graphmodule, which creates
        # placeholders for external inputs in argument order. Since mul's first
        # arg is the conv output (user input) and the second is the constant,
        # the subgraph naturally has the user input before the constant.
        sub_gm, _ = create_submodule_from_nodes(
            edge_ep.graph_module, node_list, self.TAG
        )

        # Tag the constant placeholder in the submodule, matching what
        # tag_constant_data does in the normal partitioning pipeline.
        constant_names = set(
            edge_ep.graph_signature.inputs_to_lifted_tensor_constants.keys()
        )
        for node in sub_gm.graph.nodes:
            if node.op == "placeholder" and node.name in constant_names:
                node.meta["delegation_tag"] = self.TAG

        # Verify the subgraph has user input before constant (the bug condition).
        placeholder_names = [
            n.name for n in sub_gm.graph.nodes if n.op == "placeholder"
        ]
        constant_idx = next(
            i for i, name in enumerate(placeholder_names) if name in constant_names
        )
        user_input_idx = next(
            i for i, name in enumerate(placeholder_names) if name not in constant_names
        )
        self.assertGreater(
            constant_idx,
            user_input_idx,
            "Test precondition: constant should appear after user input before fix",
        )

        # arrange_graph_placeholders should sort the constant before user inputs.
        arrange_graph_placeholders(sub_gm, edge_ep, self.TAG)

        placeholder_names = [
            n.name for n in sub_gm.graph.nodes if n.op == "placeholder"
        ]
        constant_idx = next(
            i for i, name in enumerate(placeholder_names) if name in constant_names
        )
        user_input_idx = next(
            i for i, name in enumerate(placeholder_names) if name not in constant_names
        )
        self.assertLess(
            constant_idx,
            user_input_idx,
            f"Constant should precede user input. Order: {placeholder_names}",
        )


if __name__ == "__main__":
    unittest.main()
