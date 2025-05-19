# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

# TODO Move these
from executorch.backends.xnnpack.test.tester.tester import (
    Export,
    Partition,
    Quantize,
    RunPasses,
    Serialize,
    Stage,
    ToEdge,
    ToEdgeTransformAndLower,
    ToExecutorch,
)

class TesterBase(ABC):
    @abstractmethod
    def quantize(self, quantize_stage: Optional[Quantize] = None):
        pass

    @abstractmethod
    def export(self, export_stage: Optional[Export] = None):
        pass

    @abstractmethod
    def to_edge(self, to_edge_stage: Optional[ToEdge] = None):
        pass

    @abstractmethod
    def to_edge_transform_and_lower(
        self, to_edge_and_transform_stage: Optional[ToEdgeTransformAndLower] = None
    ):
        pass

    @abstractmethod
    def run_passes(self, run_passes_stage: Optional[RunPasses] = None):
        pass

    @abstractmethod
    def partition(self, partition_stage: Optional[Partition] = None):
        pass

    @abstractmethod
    def to_executorch(self, to_executorch_stage: Optional[ToExecutorch] = None):
        pass

    @abstractmethod
    def serialize(self, serialize_stage: Optional[Serialize] = None):
        pass

    @abstractmethod
    def dump_artifact(self, path: Optional[str] = None, stage: Optional[str] = None):
        pass

    @abstractmethod
    def get_artifact(self, stage: Optional[str] = None):
        pass

    @abstractmethod
    def check(self, input: List[str]):
        pass

    @abstractmethod
    def check_not(self, input: List[str]):
        pass

    @abstractmethod
    def check_count(self, input: Dict[Any, int]):
        pass

    @abstractmethod
    def check_node_count(self, input: Dict[Any, int]):
        pass

    @abstractmethod
    def visualize(
        self, reuse_server: bool = True, stage: Optional[str] = None, **kwargs
    ):
        pass

    def run_method_and_compare_outputs(
        self,
        stage: Optional[str] = None,
        inputs: Optional[Tuple[torch.Tensor]] = None,
        num_runs=1,
        atol=1e-03,
        rtol=1e-03,
        qtol=0,
    ):
        pass
