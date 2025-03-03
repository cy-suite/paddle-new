// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
// #include <torch/types.h>
#include <tuple>
#include <vector>
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/types.h"

#include "paddle/fluid/distributed/collective/deep_ep/config.hpp"
#include "paddle/fluid/distributed/collective/deep_ep/event.hpp"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/exception.cuh"
#include "paddle/phi/api/include/tensor.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"

namespace deep_ep {

struct Buffer {
  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8,
                   "The number of maximum NVLink peers must be 8");

 private:
  // Low-latency mode buffer
  int low_latency_buffer_idx = 0;
  bool low_latency_mode = false;

  // NVLink Buffer
  int64_t num_nvl_bytes;
  void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  void** buffer_ptrs_gpu = nullptr;

  // NVSHMEM Buffer
  int64_t num_rdma_bytes;
  void* rdma_buffer_ptr = nullptr;

  // Device info and communication
  int device_id;
  int rank, rdma_rank, nvl_rank;
  int num_ranks, num_rdma_ranks, num_nvl_ranks;
  cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

  // Stream for communication
  // c10::cuda::CUDAStream comm_stream;
  cudaStream_t comm_stream;
  phi::distributed::NCCLCommContext* comm_ctx;
  phi::GPUContext* calc_ctx;

  // After IPC/NVSHMEM synchronization, this flag will be true
  bool available = false;

  // Task fifo
  int head = 0;
  int* task_fifo_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  int** task_fifo_ptrs_gpu = nullptr;

  // Workspace
  void* workspace = nullptr;

  // Host-side MoE info
  volatile int* moe_recv_counter = nullptr;
  int* moe_recv_counter_mapped = nullptr;

  // Host-side expert-level MoE info
  volatile int* moe_recv_expert_counter = nullptr;
  int* moe_recv_expert_counter_mapped = nullptr;

  // Host-side RDMA-level MoE info
  volatile int* moe_recv_rdma_counter = nullptr;
  int* moe_recv_rdma_counter_mapped = nullptr;

 private:
  void move_fifo_slots(int num_slots = 1);

 public:
  Buffer(int rank,
         int num_ranks,
         int64_t num_nvl_bytes,
         int64_t num_rdma_bytes,
         bool low_latency_mode,
         int context_ring_id);

  ~Buffer() noexcept(false);

  bool is_available() const;

  bool is_internode_available() const;

  int get_num_rdma_ranks() const;

  int get_rdma_rank() const;

  int get_root_rdma_rank(bool global) const;

  int get_local_device_id() const;

  pybind11::bytearray get_local_ipc_handle() const;

  // pybind11::bytearray get_local_nvshmem_unique_id() const;

  // torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype,
  // int64_t offset, bool use_rdma_buffer) const;

  void sync(const std::vector<int>& device_ids,
            const std::vector<std::optional<pybind11::bytearray>>&
                all_gathered_handles,
            const std::optional<pybind11::bytearray>& root_unique_id_opt);

  std::tuple<torch::Tensor,
             std::optional<torch::Tensor>,
             torch::Tensor,
             torch::Tensor,
             std::optional<EventHandle>>
  get_dispatch_layout(const torch::Tensor& topk_idx,
                      int num_experts,
                      std::optional<EventHandle>& previous_event,  // NOLINT
                      bool async,
                      bool allocate_on_comm_stream);

  std::tuple<torch::Tensor,
             std::optional<torch::Tensor>,
             std::optional<torch::Tensor>,
             std::optional<torch::Tensor>,
             std::vector<int>,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             std::optional<EventHandle>>
  intranode_dispatch(
      const torch::Tensor& x,
      const std::optional<torch::Tensor>& x_scales,
      const std::optional<torch::Tensor>& topk_idx,
      const std::optional<torch::Tensor>& topk_weights,
      const std::optional<torch::Tensor>& num_tokens_per_rank,
      const torch::Tensor& is_token_in_rank,
      const std::optional<torch::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
      const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<torch::Tensor,
             std::optional<torch::Tensor>,
             std::optional<EventHandle>>
  intranode_combine(const torch::Tensor& x,
                    const std::optional<torch::Tensor>& topk_weights,
                    const torch::Tensor& src_idx,
                    const torch::Tensor& rank_prefix_matrix,
                    const torch::Tensor& channel_prefix_matrix,
                    const torch::Tensor& send_head,
                    const Config& config,
                    std::optional<EventHandle>& previous_event,  // NOLINT
                    bool async,
                    bool allocate_on_comm_stream);

  std::tuple<torch::Tensor,
             std::optional<torch::Tensor>,
             std::optional<torch::Tensor>,
             std::optional<torch::Tensor>,
             std::vector<int>,
             torch::Tensor,
             torch::Tensor,
             std::optional<torch::Tensor>,
             torch::Tensor,
             std::optional<torch::Tensor>,
             torch::Tensor,
             std::optional<torch::Tensor>,
             std::optional<torch::Tensor>,
             std::optional<torch::Tensor>,
             std::optional<EventHandle>>
  internode_dispatch(
      const torch::Tensor& x,
      const std::optional<torch::Tensor>& x_scales,
      const std::optional<torch::Tensor>& topk_idx,
      const std::optional<torch::Tensor>& topk_weights,
      const std::optional<torch::Tensor>& num_tokens_per_rank,
      const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
      const torch::Tensor& is_token_in_rank,
      const std::optional<torch::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      int cached_num_rdma_recv_tokens,
      const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
      const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
      const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
      const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<torch::Tensor,
             std::optional<torch::Tensor>,
             std::optional<EventHandle>>
  internode_combine(const torch::Tensor& x,
                    const std::optional<torch::Tensor>& topk_weights,
                    const torch::Tensor& src_meta,
                    const torch::Tensor& is_combined_token_in_rank,
                    const torch::Tensor& rdma_channel_prefix_matrix,
                    const torch::Tensor& rdma_rank_prefix_sum,
                    const torch::Tensor& gbl_channel_prefix_matrix,
                    const torch::Tensor& combined_rdma_head,
                    const torch::Tensor& combined_nvl_head,
                    const Config& config,
                    std::optional<EventHandle>& previous_event,  // NOLINT
                    bool async,
                    bool allocate_on_comm_stream);

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank,
                                int hidden,
                                int num_experts);

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_dispatch(const torch::Tensor& x,
                       const torch::Tensor& topk_idx,
                       int num_max_dispatch_tokens_per_rank,
                       int num_experts,
                       bool async,
                       bool return_recv_hook);

  std::tuple<torch::Tensor,
             std::optional<EventHandle>,
             std::optional<std::function<void()>>>
  low_latency_combine(const torch::Tensor& x,
                      const torch::Tensor& topk_idx,
                      const torch::Tensor& topk_weights,
                      const torch::Tensor& src_info,
                      const torch::Tensor& layout_range,
                      int num_max_dispatch_tokens_per_rank,
                      int num_experts,
                      bool async,
                      bool return_recv_hook);

  // std::tuple<paddle::Tensor, std::optional<paddle::Tensor>,
  // std::optional<paddle::Tensor>, std::optional<paddle::Tensor>,
  // std::vector<int>, paddle::Tensor, paddle::Tensor,
  // std::optional<paddle::Tensor>, paddle::Tensor,
  // std::optional<paddle::Tensor>, paddle::Tensor,
  // std::optional<paddle::Tensor>, std::optional<paddle::Tensor>,
  // std::optional<paddle::Tensor>, std::optional<EventHandle>>
  // internode_dispatch(const paddle::Tensor& x, const
  // std::optional<paddle::Tensor>& x_scales,
  //                    const std::optional<paddle::Tensor>& topk_idx, const
  //                    std::optional<paddle::Tensor>& topk_weights, const
  //                    std::optional<paddle::Tensor>& num_tokens_per_rank,
  //                    const std::optional<paddle::Tensor>&
  //                    num_tokens_per_rdma_rank, const paddle::Tensor&
  //                    is_token_in_rank, const std::optional<paddle::Tensor>&
  //                    num_tokens_per_expert, int cached_num_recv_tokens, int
  //                    cached_num_rdma_recv_tokens, const
  //                    std::optional<paddle::Tensor>&
  //                    cached_rdma_channel_prefix_matrix, const
  //                    std::optional<paddle::Tensor>&
  //                    cached_recv_rdma_rank_prefix_sum, const
  //                    std::optional<paddle::Tensor>&
  //                    cached_gbl_channel_prefix_matrix, const
  //                    std::optional<paddle::Tensor>&
  //                    cached_recv_gbl_rank_prefix_sum, int expert_alignment,
  //                    const Config& config, std::optional<EventHandle>&
  //                    previous_event, bool async, bool
  //                    allocate_on_comm_stream);

  // std::tuple<paddle::Tensor, std::optional<paddle::Tensor>,
  // std::optional<EventHandle>> internode_combine(const paddle::Tensor& x,
  // const std::optional<paddle::Tensor>& topk_weights,
  //                   const paddle::Tensor& src_meta, const paddle::Tensor&
  //                   is_combined_token_in_rank, const paddle::Tensor&
  //                   rdma_channel_prefix_matrix, const paddle::Tensor&
  //                   rdma_rank_prefix_sum, const paddle::Tensor&
  //                   gbl_channel_prefix_matrix, const paddle::Tensor&
  //                   combined_rdma_head, const paddle::Tensor&
  //                   combined_nvl_head, const Config& config,
  //                   std::optional<EventHandle>& previous_event, bool async,
  //                   bool allocate_on_comm_stream);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             paddle::Tensor,
             paddle::Tensor,
             std::optional<EventHandle>>
  get_dispatch_layout_api(const paddle::Tensor& topk_idx,
                          int num_experts,
                          std::optional<EventHandle>& previous_event,  // NOLINT
                          bool async,
                          bool allocate_on_comm_stream);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::optional<paddle::Tensor>,
             std::vector<int>,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             paddle::Tensor,
             std::optional<EventHandle>>
  intranode_dispatch_api(
      const paddle::Tensor& x,
      const std::optional<paddle::Tensor>& x_scales,
      const std::optional<paddle::Tensor>& topk_idx,
      const std::optional<paddle::Tensor>& topk_weights,
      const std::optional<paddle::Tensor>& num_tokens_per_rank,
      const paddle::Tensor& is_token_in_rank,
      const std::optional<paddle::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      const std::optional<paddle::Tensor>& cached_rank_prefix_matrix,
      const std::optional<paddle::Tensor>& cached_channel_prefix_matrix,
      int expert_alignment,
      const Config& config,
      std::optional<EventHandle>& previous_event,  // NOLINT
      bool async,
      bool allocate_on_comm_stream);

  std::tuple<paddle::Tensor,
             std::optional<paddle::Tensor>,
             std::optional<EventHandle>>
  intranode_combine_api(const paddle::Tensor& x,
                        const std::optional<paddle::Tensor>& topk_weights,
                        const paddle::Tensor& src_idx,
                        const paddle::Tensor& rank_prefix_matrix,
                        const paddle::Tensor& channel_prefix_matrix,
                        const paddle::Tensor& send_head,
                        const Config& config,
                        std::optional<EventHandle>& previous_event,  // NOLINT
                        bool async,
                        bool allocate_on_comm_stream);
};

torch::Tensor ConvertPaddleTensorToFakeTorchTensor(
    const paddle::Tensor& tensor);
paddle::Tensor ConvertFakeTorchTensorToPaddleTensor(
    const torch::Tensor& tensor);

}  // namespace deep_ep
