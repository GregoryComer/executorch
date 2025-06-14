from typing import Any, List, Optional, Sequence, Type, Tuple

import torch

from executorch.backends.test.harness.stages.stage import Stage, StageType
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower
)
from executorch.exir.backend.partitioner import Partitioner
from torch.export import ExportedProgram

class ToEdgeTransformAndLower(Stage):
    def __init__(
        self,
        default_partitioner_cls: Type,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        self.partitioners = partitioners or ([default_partitioner_cls()] if default_partitioner_cls is not None else None)
        self.edge_compile_conf = (
            edge_compile_config or EdgeCompileConfig()
        )
        self.edge_dialect_program = None
    
    def stage_type(self) -> StageType:
        return StageType.TO_EDGE_TRANSFORM_AND_LOWER

    def run(self, artifact: ExportedProgram, inputs=None) -> None:
        self.edge_dialect_program = to_edge_transform_and_lower(
            artifact,
            compile_config=self.edge_compile_conf,
            partitioner=self.partitioners,
        )

    @property
    def artifact(self) -> EdgeProgramManager:
        return self.edge_dialect_program

    @property
    def graph_module(self) -> str:
        return self.edge_dialect_program.exported_program().graph_module
