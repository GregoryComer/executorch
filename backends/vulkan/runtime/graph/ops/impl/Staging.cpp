/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

namespace vkcompute {

void add_staging_to_tensor_node(
    ComputeGraph& graph,
    const ValueRef in_staging,
    const ValueRef out_tensor) {
  VK_CHECK_COND(graph.val_is_staging(in_staging));

  vkapi::ShaderInfo shader = get_nchw_to_tensor_shader(
      *graph.get_tensor(out_tensor), graph.int8_buffers_enabled());

  std::vector<PushConstantDataInfo> pcs;
  if (graph.is_buffer_storage(out_tensor)) {
    pcs = {
        graph.sizes_pc_of(out_tensor),
        graph.strides_pc_of(out_tensor),
        graph.numel_pc_of(out_tensor)};
  } else {
    pcs = {graph.sizes_pc_of(out_tensor)};
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      shader,
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Input and Outputs
      {{out_tensor, vkapi::kWrite}, {in_staging, vkapi::kRead}},
      // Parameter Buffers
      {},
      // Push Constants
      pcs,
      // Specialization Constants
      {graph.hashed_layout_of(out_tensor)},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

const std::string kBitw8PrefixStr = "bitw8_image_to_nchw_nobitw8buffer";

bool is_bitw8_shader(const vkapi::ShaderInfo& shader) {
  const auto size = kBitw8PrefixStr.size();
  const std::string& shader_prefix_str = shader.kernel_name.substr(0, size);
  return shader_prefix_str == kBitw8PrefixStr;
}

vkapi::ShaderInfo get_tensor_to_staging_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef in_tensor = args.at(1).refs.at(0);
  return get_tensor_to_nchw_shader(
      *graph->get_tensor(in_tensor), graph->int8_buffers_enabled());
}

utils::uvec3 tensor_to_staging_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef in_tensor = args.at(1).refs.at(0);
  const ValueRef out_staging = args.at(0).refs.at(0);

  utils::uvec3 global_wg_size = graph->create_global_wg_size(in_tensor);

  // Normally, the image_to_nchw shader is structured so that each thread reads
  // one texel from the input texture and writes each component of the texel
  // into the corresponding location in the output buffer. However, this shader
  // is structured slightly differently in that each thread writes out a
  // complete 32 bit integer (containing 4 packed 8-bit integers) into the
  // output buffer. Therefore, the global work group size for this shader will
  // be the number of elements in the output buffer divided by 4, as opposed to
  // the extents of the input texture.
  if (is_bitw8_shader(shader)) {
    const uint32_t buffer_len = utils::safe_downcast<uint32_t>(
        graph->get_staging(out_staging)->numel() / 4);
    global_wg_size = {buffer_len, 1, 1};
  }

  return global_wg_size;
}

void add_tensor_to_staging_node(
    ComputeGraph& graph,
    const ValueRef in_tensor,
    const ValueRef out_staging) {
  VK_CHECK_COND(graph.val_is_staging(out_staging));

  vkapi::ShaderInfo shader = get_tensor_to_nchw_shader(
      *graph.get_tensor(in_tensor), graph.int8_buffers_enabled());

  std::vector<PushConstantDataInfo> pcs;
  if (graph.is_buffer_storage(in_tensor)) {
    pcs = {
        graph.sizes_pc_of(in_tensor),
        graph.strides_pc_of(in_tensor),
        graph.numel_pc_of(in_tensor)};
  } else {
    pcs = {graph.sizes_pc_of(in_tensor)};
  }

  if (is_bitw8_shader(shader)) {
    pcs.push_back(graph.numel_pc_of(in_tensor));
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      shader,
      tensor_to_staging_global_wg_size,
      default_pick_local_wg_size,
      // Input and Outputs
      {{out_staging, vkapi::kWrite}, {in_tensor, vkapi::kRead}},
      // Parameter Buffers
      {},
      // Push Constants
      pcs,
      // Specialization Constants
      {graph.hashed_layout_of(in_tensor)},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void add_prepack_standard_node(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef tensor,
    const bool transpose_hw = false) {
  vkapi::ShaderInfo shader = get_nchw_to_tensor_shader(
      *graph.get_tensor(tensor), graph.int8_buffers_enabled());

  std::vector<PushConstantDataInfo> pcs;
  if (graph.is_buffer_storage(tensor)) {
    pcs = {
        graph.sizes_pc_of(tensor),
        graph.strides_pc_of(tensor),
        graph.numel_pc_of(tensor)};
  } else {
    pcs = {graph.sizes_pc_of(tensor)};
  }

  int transpose_hw_spec = transpose_hw ? 1 : 0;

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_global_wg_size(tensor),
      graph.create_local_wg_size(tensor),
      // Input and Outputs
      tensor_data,
      tensor,
      // Parameter Buffers
      {},
      // Specialization Constants
      {graph.hashed_layout_of(tensor), transpose_hw_spec},
      pcs));
}

ValueRef prepack_standard(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout layout,
    const bool passthrough,
    const utils::AxisMapLayout axis_map_layout) {
  if (passthrough && graph.val_is_tensor(tensor_data)) {
    return tensor_data;
  }
  VK_CHECK_COND(graph.val_is_tref(tensor_data));
  ValueRef tensor =
      graph.add_tensor_like(tensor_data, storage_type, layout, axis_map_layout);
  add_prepack_standard_node(graph, tensor_data, tensor);
  return tensor;
}

ValueRef prepack_standard_hw_transposed(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout layout,
    const bool passthrough,
    const utils::AxisMapLayout axis_map_layout) {
  (void)passthrough;

  VK_CHECK_COND(graph.val_is_tref(tensor_data));
  std::vector<int64_t> new_out_sizes = graph.sizes_of(tensor_data);
  const int w_dim = new_out_sizes.size() - 1;
  const int h_dim = new_out_sizes.size() - 2;
  const int64_t tmp = new_out_sizes.at(w_dim);
  new_out_sizes.at(w_dim) = new_out_sizes.at(h_dim);
  new_out_sizes.at(h_dim) = tmp;
  ValueRef tensor = graph.add_tensor(
      new_out_sizes,
      graph.dtype_of(tensor_data),
      storage_type,
      layout,
      -1,
      axis_map_layout);
  add_prepack_standard_node(graph, tensor_data, tensor, true);
  return tensor;
}

ValueRef prepack_standard_like(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef to_copy,
    const bool passthrough) {
  VK_CHECK_COND(graph.val_is_tensor(to_copy));
  return prepack_standard(
      graph,
      tensor_data,
      graph.storage_type_of(to_copy),
      graph.estimate_memory_layout_of(to_copy),
      passthrough);
}

void add_prepack_direct_copy_buffer_node(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef tensor) {
  std::string kernel_name = "buffer_to_buffer";
  add_dtype_suffix(kernel_name, graph.dtype_of(tensor_data));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList ubos;
  ubos.append({graph.numel_ubo(tensor)});

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_global_wg_size(tensor),
      graph.create_local_wg_size(tensor),
      // Input and Outputs
      tensor_data,
      tensor,
      // Parameter Buffers
      ubos,
      // Specialization Constants
      {}));
}

ValueRef prepack_direct_copy_buffer(
    ComputeGraph& graph,
    const ValueRef tensor_data) {
  VK_CHECK_COND(graph.val_is_tref(tensor_data));
  ValueRef tensor =
      graph.add_tensor_like(tensor_data, utils::kBuffer, utils::kWidthPacked);
  add_prepack_direct_copy_buffer_node(graph, tensor_data, tensor);
  return tensor;
}

ValueRef prepack_int4_linear_weight_transposed_interleaved(
    ComputeGraph& graph,
    const ValueRef qmat2_data) {
  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  const int64_t K = qmat2_orig_sizes.at(ndim - 1) * 2;
  const int64_t N = qmat2_orig_sizes.at(ndim - 2);
  const int64_t N_div2 = N / int64_t(2);

  utils::StorageType storage_type = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (N_div2 > max_extent * 4 || K > max_extent) {
    storage_type = utils::kBuffer;
  }

  std::vector<int64_t> qmat2_sizes{K, N_div2};
  ValueRef qmat2 = graph.add_tensor(
      qmat2_sizes, vkcompute::vkapi::kByte, storage_type, utils::kWidthPacked);

  utils::uvec3 global_wg_size;
  global_wg_size = graph.logical_limits_of(qmat2);
  global_wg_size[1] = utils::div_up(global_wg_size[1], uint32_t(2));

  std::string kernel_name =
      graph.context()->adapter_ptr()->has_full_int8_buffers_support()
      ? "pack_int4_linear_weight_transposed_interleaved"
      : "pack_int4_linear_weight_transposed_interleaved_nobitw8buffer";
  add_storage_type_suffix(kernel_name, storage_type);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Inputs and Outputs
      qmat2_data,
      qmat2,
      // UBOs
      {},
      // Specialization Constants
      {},
      // Push Constants
      {graph.sizes_pc_of(qmat2)}));

  return qmat2;
}

ValueRef prepack_int4_linear_weight_transposed_block_4x8(
    ComputeGraph& graph,
    const ValueRef qmat2_data) {
  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  const int64_t K_div2 = qmat2_orig_sizes.at(ndim - 1); // Input is [N, K/2]
  const int64_t N = qmat2_orig_sizes.at(ndim - 2);
  // Logical K dimension. Each value in the tensor is a uint8 that contains 2
  // packed 4-bit values.
  const int64_t K = K_div2 * 2;

  // This packing format partitions the weight tensor into 4 wide x 8 high
  // blocks. To figure out the size of the output tensor, determine the number
  // of blocks along the width and height dims.
  const int64_t num_blocks_K = utils::div_up(K, int64_t(4));
  const int64_t num_blocks_N = utils::div_up(N, int64_t(8));
  // Each transposed block is 8 wide x 4 high. In terms of 8-bit values, the
  // block is 4 wide x 4 high. To maximize memory loading efficiency, the packed
  // weight tensor will use a base data type of uint32_t; in terms of uint32_t,
  // each block is 1 wide x 4 high. However, each block is also flattened as it
  // is stored, so that the whole block can be loaded at once. As a result, the
  // stored block will be 4 wide x 1 high.
  const int64_t output_width = num_blocks_K * 4;
  const int64_t output_height = num_blocks_N;

  // Store the original sizes of the tensor to pass to the shader
  utils::ivec2 orig_sizes{
      utils::safe_downcast<int32_t>(K), utils::safe_downcast<int32_t>(N)};

  std::vector<int64_t> qmat2_sizes{output_height, output_width};

  utils::StorageType storage_type = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (output_width > max_extent * 4 || output_height > max_extent) {
    storage_type = utils::kBuffer;
  }

  ValueRef qmat2 = graph.add_tensor(
      qmat2_sizes, vkcompute::vkapi::kUInt, storage_type, utils::kWidthPacked);

  // Global workgroup size: each thread writes out two adjacent blocks
  utils::uvec3 global_wg_size{
      utils::div_up(utils::safe_downcast<uint32_t>(num_blocks_K), uint32_t(2)),
      utils::safe_downcast<uint32_t>(num_blocks_N),
      1u};

  std::string kernel_name = "pack_int4_linear_weight_transposed_block_4x8";
  add_storage_type_suffix(kernel_name, storage_type);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Inputs and Outputs
      qmat2_data,
      qmat2,
      // UBOs
      {},
      // Specialization Constants
      {},
      // Push Constants
      {graph.sizes_pc_of(qmat2),
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec2))}));

  return qmat2;
}

void prepack_op(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_prepack_standard_node(graph, args[0], args[1]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.prepack.default, prepack_op);
}

} // namespace vkcompute
