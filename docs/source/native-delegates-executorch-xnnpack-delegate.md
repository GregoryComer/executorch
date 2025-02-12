# XNNPACK Backend

This is a high-level overview of the ExecuTorch XNNPACK backend delegate. This high performance delegate is aimed to reduce CPU inference latency for ExecuTorch models. We will provide a brief introduction to the XNNPACK library and explore the delegate’s overall architecture and intended use cases.

## What is XNNPACK?
XNNPACK is a library of highly-optimized neural network operators for ARM, x86, and WebAssembly architectures in Android, iOS, Windows, Linux, and macOS environments. It is an open source project, you can find more information about it on [github](https://github.com/google/XNNPACK).

## What are ExecuTorch delegates?
A delegate is an entry point for backends to process and execute parts of the ExecuTorch program. Delegated portions of ExecuTorch models hand off execution to backends. The XNNPACK backend delegate is one of many available in ExecuTorch. It leverages the XNNPACK third-party library to accelerate ExecuTorch programs efficiently across a variety of CPUs. More detailed information on the delegates and developing your own delegates is available [here](compiler-delegate-and-partitioner.md). It is recommended that you get familiar with that content before continuing on to the Architecture section.

## Architecture
![High Level XNNPACK delegate Architecture](./xnnpack-delegate-architecture.png)

### Ahead-of-time
In the ExecuTorch export flow, lowering to the XNNPACK delegate happens at the `to_backend()` stage. In this stage, the model is partitioned by the `XnnpackPartitioner`. Partitioned sections of the graph are converted to a XNNPACK specific graph represenationed and then serialized via flatbuffer. The serialized flatbuffer is then ready to be deserialized and executed by the XNNPACK backend at runtime.

![ExecuTorch XNNPACK delegate Export Flow](./xnnpack-et-flow-diagram.png)

#### Partitioner
The partitioner is implemented by backend delegates to mark nodes suitable for lowering. The `XnnpackPartitioner` lowers using node targets and module metadata. Some more references for partitioners can be found [here](compiler-delegate-and-partitioner.md)

##### Module-based partitioning

`source_fn_stack` is embedded in the node’s metadata and gives information on where these nodes come from. For example, modules like `torch.nn.Linear` when captured and exported `to_edge` generate groups of nodes for their computation. The group of nodes associated with computing the linear module then has a `source_fn_stack` of `torch.nn.Linear. Partitioning based on `source_fn_stack` allows us to identify groups of nodes which are lowerable via XNNPACK.

For example after capturing `torch.nn.Linear` you would find the following key in the metadata for the addmm node associated with linear:
```python
>>> print(linear_node.meta["source_fn_stack"])
'source_fn_stack': ('fn', <class 'torch.nn.modules.linear.Linear'>)
```


##### Op-based partitioning

The `XnnpackPartitioner` also partitions using op targets. It traverses the graph and identifies individual nodes which are lowerable to XNNPACK. A drawback to module-based partitioning is that operators which come from [decompositions](https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py) may be skipped. For example, an operator like `torch.nn.Hardsigmoid` is decomposed into add, muls, divs, and clamps. While hardsigmoid is not lowerable, we can lower the decomposed ops. Relying on `source_fn_stack` metadata would skip these lowerables because they belong to a non-lowerable module, so in order to improve model performance, we greedily lower operators based on the op targets as well as the `source_fn_stack`.

##### Passes

Before any serialization, we apply passes on the subgraphs to prepare the graph. These passes are essentially graph transformations that help improve the performance of the delegate. We give an overview of the most significant passes and their function below. For a description of all passes see [here](https://github.com/pytorch/executorch/tree/main/backends/xnnpack/_passes):

* Channels Last Reshape
    * ExecuTorch tensors tend to be contiguous before passing them into delegates, while XNNPACK only accepts channels-last memory layout. This pass minimizes the number of permutation operators inserted to pass in channels-last memory format.
* Conv1d to Conv2d
    * Allows us to delegate Conv1d nodes by transforming them to Conv2d
* Conv and BN Fusion
    * Fuses batch norm operations with the previous convolution node

#### Serialiazation
After partitioning the lowerable subgraphs from the model, The XNNPACK delegate pre-processes these subgraphs and serializes them via flatbuffer for the XNNPACK backend.


##### Serialization Schema

The XNNPACK delegate uses flatbuffer for serialization. In order to improve runtime performance, the XNNPACK delegate’s flatbuffer [schema](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/serialization/schema.fbs) mirrors the XNNPACK Library’s graph level API calls. The serialized data are arguments to XNNPACK’s APIs, so that at runtime, the XNNPACK execution graph can efficiently be created with successive calls to XNNPACK’s APIs.

### Runtime
The XNNPACK backend’s runtime interfaces with the ExecuTorch runtime through the custom `init` and `execute` function. Each delegated subgraph is contained in an individually serialized XNNPACK blob. When the model is initialized, ExecuTorch calls `init` on all XNNPACK Blobs to load the subgraph from serialized flatbuffer. After, when the model is executed, each subgraph is executed via the backend through the custom `execute` function. To read more about how delegate runtimes interface with ExecuTorch, refer to this [resource](compiler-delegate-and-partitioner.md).


#### **XNNPACK Library**
XNNPACK delegate supports CPU's on multiple platforms; more information on the supported hardware architectures can be found on the XNNPACK Library’s [README](https://github.com/google/XNNPACK).

#### **Init**
When calling XNNPACK delegate’s `init`, we deserialize the preprocessed blobs via flatbuffer. We define the nodes (operators) and edges (intermediate tensors) to build the XNNPACK execution graph using the information we serialized ahead-of-time. As we mentioned earlier, the majority of processing has been done ahead-of-time, so that at runtime we can just call the XNNPACK APIs with the serialized arguments in succession. As we define static data into the execution graph, XNNPACK performs weight packing at runtime to prepare static data like weights and biases for efficient execution. After creating the execution graph, we create the runtime object and pass it on to `execute`.

Since weight packing creates an extra copy of the weights inside XNNPACK, We free the original copy of the weights inside the preprocessed XNNPACK Blob, this allows us to remove some of the memory overhead.


#### **Execute**
When executing the XNNPACK subgraphs, we prepare the tensor inputs and outputs and feed them to the XNNPACK runtime graph. After executing the runtime graph, the output pointers are filled with the computed tensors.

#### **Profiling**
We have enabled basic profiling for the XNNPACK delegate that can be enabled with the compiler flag `-DEXECUTORCH_ENABLE_EVENT_TRACER` (add `-DENABLE_XNNPACK_PROFILING` for additional details). With ExecuTorch's Developer Tools integration, you can also now use the Developer Tools to profile the model. You can follow the steps in [Using the ExecuTorch Developer Tools to Profile a Model](./tutorials/devtools-integration-tutorial) on how to profile ExecuTorch models and use Developer Tools' Inspector API to view XNNPACK's internal profiling information. An example implementation is available in the `xnn_executor_runner` (see [tutorial here](tutorial-xnnpack-delegate-lowering.md#profiling)).


[comment]: <> (TODO: Refactor quantizer to a more official quantization doc)
## Quantization
The XNNPACK delegate can also be used as a backend to execute symmetrically quantized models. For quantized model delegation, we quantize models using the `XNNPACKQuantizer`. `Quantizers` are backend specific, which means the `XNNPACKQuantizer` is configured to quantize models to leverage the quantized operators offered by the XNNPACK Library. We will not go over the details of how to implement your custom quantizer, you can follow the docs [here](https://pytorch.org/tutorials/prototype/pt2e_quantizer.html) to do so. However, we will provide a brief overview of how to quantize the model to leverage quantized execution of the XNNPACK delegate.

### Configuring the XNNPACKQuantizer

```python
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
```
Here we initialize the `XNNPACKQuantizer` and set the quantization config to be symmetrically quantized. Symmetric quantization is when weights are symmetrically quantized with `qmin = -127` and `qmax = 127`, which forces the quantization zeropoints to be zero. `get_symmetric_quantization_config()` can be configured with the following arguments:
* `is_per_channel`
    * Weights are quantized across channels
* `is_qat`
    * Quantize aware training
* `is_dynamic`
    * Dynamic quantization

We can then configure the `XNNPACKQuantizer` as we wish. We set the following configs below as an example:
```python
quantizer.set_global(quantization_config)
    .set_object_type(torch.nn.Conv2d, quantization_config) # can configure by module type
    .set_object_type(torch.nn.functional.linear, quantization_config) # or torch functional op typea
    .set_module_name("foo.bar", quantization_config)  # or by module fully qualified name
```

### Quantizing your model with the XNNPACKQuantizer
After configuring our quantizer, we are now ready to quantize our model
```python
from torch.export import export_for_training

exported_model = export_for_training(model_to_quantize, example_inputs).module()
prepared_model = prepare_pt2e(exported_model, quantizer)
print(prepared_model.graph)
```
Prepare performs some Conv2d-BN fusion, and inserts quantization observers in the appropriate places. For Post-Training Quantization, we generally calibrate our model after this step. We run sample examples through the `prepared_model` to observe the statistics of the Tensors to calculate the quantization parameters.

Finally, we convert our model here:
```python
quantized_model = convert_pt2e(prepared_model)
print(quantized_model)
```
You will now see the Q/DQ representation of the model, which means `torch.ops.quantized_decomposed.dequantize_per_tensor` are inserted at quantized operator inputs and `torch.ops.quantized_decomposed.quantize_per_tensor` are inserted at operator outputs. Example:

```python
def _qdq_quantized_linear(
    x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max,
    weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max,
    bias_fp32,
    out_scale, out_zero_point, out_quant_min, out_quant_max
):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    out_fp32 = torch.ops.aten.linear.default(x_fp32, weight_fp32, bias_fp32)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
    return out_i8
```


You can read more indepth explanations on PyTorch 2 quantization [here](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html).
K
```python
import torch
import torchvision.models as models

from torch.export import export, ExportedProgram
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower
from executorch.exir.backend.backend_api import to_backend


mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

exported_program: ExportedProgram = export(mobilenet_v2, sample_inputs)
edge: EdgeProgramManager = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
)
```

## Lowering a model to XNNPACK

We will go through this example with the [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) pretrained model downloaded from the TorchVision library. The flow of lowering a model starts after exporting the model `to_edge`. We call the `to_backend` api with the `XnnpackPartitioner`. The partitioner identifies the subgraphs suitable for XNNPACK backend delegate to consume. Afterwards, the identified subgraphs will be serialized with the XNNPACK Delegate flatbuffer schema and each subgraph will be replaced with a call to the XNNPACK Delegate.

```python
>>> print(edge.exported_program().graph_module)
GraphModule(
  (lowered_module_0): LoweredBackendModule()
  (lowered_module_1): LoweredBackendModule()
)



def forward(self, b_features_0_1_num_batches_tracked, ..., x):
    lowered_module_0 = self.lowered_module_0
    lowered_module_1 = self.lowered_module_1
    executorch_call_delegate_1 = torch.ops.higher_order.executorch_call_delegate(lowered_module_1, x);  lowered_module_1 = x = None
    getitem_53 = executorch_call_delegate_1[0];  executorch_call_delegate_1 = None
    aten_view_copy_default = executorch_exir_dialects_edge__ops_aten_view_copy_default(getitem_53, [1, 1280]);  getitem_53 = None
    aten_clone_default = executorch_exir_dialects_edge__ops_aten_clone_default(aten_view_copy_default);  aten_view_copy_default = None
    executorch_call_delegate = torch.ops.higher_order.executorch_call_delegate(lowered_module_0, aten_clone_default);  lowered_module_0 = aten_clone_default = None
    getitem_52 = executorch_call_delegate[0];  executorch_call_delegate = None
    return (getitem_52,)
```

We print the graph after lowering above to show the new nodes that were inserted to call the XNNPACK Delegate. The subgraphs which are being delegated to XNNPACK are the first argument at each call site. It can be observed that the majority of `convolution-relu-add` blocks and `linear` blocks were able to be delegated to XNNPACK. We can also see the operators which were not able to be lowered to the XNNPACK delegate, such as `clone` and `view_copy`.

```python
exec_prog = edge.to_executorch()

with open("xnnpack_mobilenetv2.pte", "wb") as file:
    exec_prog.write_to_file(file)
```
After lowering to the XNNPACK Program, we can then prepare it for executorch and save the model as a `.pte` file. `.pte` is a binary format that stores the serialized ExecuTorch graph.


## Lowering a Quantized Model to XNNPACK
The XNNPACK delegate can also execute symmetrically quantized models. To understand the quantization flow and learn how to quantize models, refer to [Custom Quantization](quantization-custom-quantization.md) note. For the sake of this tutorial, we will leverage the `quantize()` python helper function conveniently added to the `executorch/executorch/examples` folder.

```python
from torch.export import export_for_training
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

mobilenet_v2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

mobilenet_v2 = export_for_training(mobilenet_v2, sample_inputs).module() # 2-stage export for quantization path

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)


def quantize(model, example_inputs):
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    print(f"Original model: {model}")
    quantizer = XNNPACKQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    print(f"Quantized model: {m}")
    # make sure we can export to flat buffer
    return m

quantized_mobilenetv2 = quantize(mobilenet_v2, sample_inputs)
```

Quantization requires a two stage export. First we use the `export_for_training` API to capture the model before giving it to `quantize` utility function. After performing the quantization step, we can now leverage the XNNPACK delegate to lower the quantized exported model graph. From here, the procedure is the same as for the non-quantized model lowering to XNNPACK.

```python
# Continued from earlier...
edge = to_edge_transform_and_lower(
    export(quantized_mobilenetv2, sample_inputs),
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
    partitioner=[XnnpackPartitioner()]
)

exec_prog = edge.to_executorch()

with open("qs8_xnnpack_mobilenetv2.pte", "wb") as file:
    exec_prog.write_to_file(file)
```

## Lowering with `aot_compiler.py` script
We have also provided a script to quickly lower and export a few example models. You can run the script to generate lowered fp32 and quantized models. This script is used simply for convenience and performs all the same steps as those listed in the previous two sections.

```
python -m examples.xnnpack.aot_compiler --model_name="mv2" --quantize --delegate
```

Note in the example above,
* the `-—model_name` specifies the model to use
* the `-—quantize` flag controls whether the model should be quantized or not
* the `-—delegate` flag controls whether we attempt to lower parts of the graph to the XNNPACK delegate.

The generated model file will be named `[model_name]_xnnpack_[qs8/fp32].pte` depending on the arguments supplied.

## Running the XNNPACK Model with CMake
After exporting the XNNPACK Delegated model, we can now try running it with example inputs using CMake. We can build and use the xnn_executor_runner, which is a sample wrapper for the ExecuTorch Runtime and XNNPACK Backend. We first begin by configuring the CMake build like such:
```bash
# cd to the root of executorch repo
cd executorch

# Get a clean cmake-out directory
./install_executorch.sh --clean
mkdir cmake-out

# Configure cmake
cmake \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-out .
```
Then you can build the runtime componenets with

```bash
cmake --build cmake-out -j9 --target install --config Release
```

Now you should be able to find the executable built at `./cmake-out/backends/xnnpack/xnn_executor_runner` you can run the executable with the model you generated as such
```bash
./cmake-out/backends/xnnpack/xnn_executor_runner --model_path=./mv2_xnnpack_fp32.pte
# or to run the quantized variant
./cmake-out/backends/xnnpack/xnn_executor_runner --model_path=./mv2_xnnpack_q8.pte
```

## Building and Linking with the XNNPACK Backend
You can build the XNNPACK backend [CMake target](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/CMakeLists.txt#L83), and link it with your application binary such as an Android or iOS application. For more information on this you may take a look at this [resource](demo-apps-android.md) next.

## Profiling
To enable profiling in the `xnn_executor_runner` pass the flags `-DEXECUTORCH_ENABLE_EVENT_TRACER=ON` and `-DEXECUTORCH_BUILD_DEVTOOLS=ON` to the build command (add `-DENABLE_XNNPACK_PROFILING=ON` for additional details). This will enable ETDump generation when running the inference and enables command line flags for profiling (see `xnn_executor_runner --help` for details).
