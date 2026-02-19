# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.passes.propagate_input_spec import propagate_input_spec
from executorch.exir.passes.propagate_input_spec import INPUT_SPEC_KEY
from executorch.exir.graph_module import get_control_flow_submodules
import copy

import torch
from torch.export.exported_program import ExportedProgram, InputKind, InputSpec
from torch.fx import GraphModule, GraphModule, Node


def remove_unused_parameters_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    Remove unused parameters from the exported program.
    """

    propagate_input_spec(ep)

    # This pass operates in two phases:
    #  * First, find all top-level graph inputs that are used.
    #  * Second, go back through and strip out unused params.

    # Find unused params. Don't ever remove top-level user inputs, just params/buffers/constants.
    removable_kinds = {InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR}

    all_params = {spec.target: spec for spec in ep.graph_signature.input_specs if spec.target is not None}
    used_params = _find_used_params_recursive(ep.graph_module)

    unused_param_targets = set(all_params.keys()) - set(used_params.keys())
    unused_params = {
        target: spec for target, spec in all_params.items()
        if target in unused_param_targets and spec.kind in removable_kinds
    }

    # Remove unused params, including in recursive control flow submodules.
    _remove_params_recursive(ep.graph_module, unused_params)

    # Update the EP signature and state dict.
    new_signature = copy.deepcopy(ep.graph_signature)
    for target, spec in unused_params.items():
        new_signature.input_specs.remove(spec)
        del ep._state_dict[target]
    ep._graph_signature = new_signature
    ep.graph_module.recompile()

    return ep


def _find_used_params_recursive(
    module: GraphModule,
) -> dict[str, InputSpec]:
    used_params: dict[str, InputSpec] = {}

    # Check direct (non-submodule) users of each placeholder node.
    for node in module.graph.nodes:
        if node.op == "placeholder":
            input_spec = node.meta.get(INPUT_SPEC_KEY, None)

            # If this placeholder is a top-level graph input and directly used, add
            # it to the set of known used params.
            if input_spec is not None and _placeholder_node_has_direct_usages(node):
                used_params[input_spec.target] = input_spec


    # Recurse into submodules.
    for _, submodule, _ in get_control_flow_submodules(module):
        used_params.update(_find_used_params_recursive(submodule))

    return used_params


def _placeholder_node_has_direct_usages(
    node: Node,
) -> bool:
    """
    Returns true if the given placeholder node is directly used in the enclosing submodule.
    Usages where the node is passed as an operand to a submodule aren't counted.
    """

    # Check each user. If it's not control flow, or it's directly used by a control flow
    # op - return true. Direct usages include cond conditions, scan xs, etc. where they
    # are used in the HOP, not just passed through to the submodule.
    for user in node.users:
        if user.target == torch.ops.higher_order.cond:
            # cond(pred, true_fn, false_fn, operands)
            if user.args[0] == node:
                return True
        elif user.target == torch.ops.higher_order.map_impl:
            # map(f, mapped_args, operands)
            if node in user.args[1]:
                return True
        elif user.target == torch.ops.higher_order.scan:
            # scan(combine_fn, init, xs, additional_args)
            if user.args[1] == node or node in user.args[2]:
                return True
        elif user.target == torch.ops.higher_order.while_loop:
            # cond, body, carried_inputs, additional_inputs
            continue # All while loop args are passed unmodified to submodules.
        else:
            # Not control flow, so it's a direct use.
            return True

    return False


def _remove_params_recursive(
    module: GraphModule,
    params_to_remove: dict[str, InputSpec],
) -> None:
    # Find placeholder nodes corresponding to the params to remove.
    placeholders_to_remove = [
        node for node in module.graph.nodes
        if node.op == "placeholder" and node.meta.get(INPUT_SPEC_KEY) is not None
        and node.meta.get(INPUT_SPEC_KEY).target in params_to_remove
    ]

    placeholders_to_remove_set = set(placeholders_to_remove)

    # Filter out removed placeholders from HOP operands
    for _, submodule, node in get_control_flow_submodules(module):
        if node.target == torch.ops.higher_order.cond:
            # cond(pred, true_fn, false_fn, operands)
            new_operands = tuple(x for x in node.args[3] if x not in placeholders_to_remove_set)
            node.args = (node.args[0], node.args[1], node.args[2], new_operands)
        elif node.target == torch.ops.higher_order.map_impl:
            # map(f, mapped_args, operands)
            new_operands = tuple(x for x in node.args[2] if x not in placeholders_to_remove_set)
            node.args = (node.args[0], node.args[1], new_operands)
        elif node.target == torch.ops.higher_order.scan:
            # scan(combine_fn, init, xs, additional_args)
            new_operands = tuple(x for x in node.args[3] if x not in placeholders_to_remove_set)
            node.args = (node.args[0], node.args[1], node.args[2], new_operands)
        elif node.target == torch.ops.higher_order.while_loop:
            # while_loop(cond, body, carried_inputs, additional_inputs)
            new_operands = tuple(x for x in node.args[3] if x not in placeholders_to_remove_set)
            node.args = (node.args[0], node.args[1], node.args[2], new_operands)

    # Remove the placeholder nodes.
    for node in placeholders_to_remove:
        module.graph.erase_node(node)
    
    module.recompile()
