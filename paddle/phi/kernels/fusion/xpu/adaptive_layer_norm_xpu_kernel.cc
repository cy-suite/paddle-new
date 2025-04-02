// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void AdaptiveLayerNormXPUKernel(const Context& ctx,
                                const DenseTensor& x,
                                const DenseTensor& scale,
                                const DenseTensor& bias,
                                const DenseTensor& tensor1,
                                const DenseTensor& tensor2,
                                int begin_norm_axis,
                                float epsilon,
                                float factor,
                                float scale_bias,
                                bool bias_after_scale,
                                DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* in_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* scale_data = reinterpret_cast<const XPUType*>(scale.data<T>());
  auto* bias_data = reinterpret_cast<const XPUType*>(bias.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));

  int r = xpu::SUCCESS;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(&ctx);
  auto x_shape = x.dims();
  int m = 1;
  int n = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    m *= x_shape[i];
  }
  for (int i = begin_norm_axis; i < x_shape.size(); i++) {
    n *= x_shape[i];
  }
  std::vector<int> x_shape_vec = common::vectorize<int>(x_shape);
  r = baidu::xpu::api::layer_norm(xpu_ctx->x_context(),
                                  in_data,
                                  out_data,
                                  m,
                                  n,
                                  epsilon,
                                  scale_data,
                                  bias_data,
                                  nullptr,
                                  nullptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
  DenseTensor scaled_out;
  scaled_out.Resize(tensor1.dims());
  ctx.template Alloc<T>(&scaled_out);
  r = baidu::xpu::api::scale(
      xpu_ctx->x_context(),
      reinterpret_cast<const XPUType*>(tensor1.data<T>()),
      reinterpret_cast<XPUType*>(scaled_out.data<T>()),
      tensor1.numel(),
      bias_after_scale,
      factor,
      scale_bias);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = baidu::xpu::api::broadcast_mul(
      xpu_ctx->x_context(),
      out_data,
      reinterpret_cast<XPUType*>(scaled_out.data<T>()),
      out_data,
      common::vectorize<int>(x.dims()),
      common::vectorize<int>(tensor1.dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

  r = baidu::xpu::api::broadcast_add(
      xpu_ctx->x_context(),
      out_data,
      reinterpret_cast<const XPUType*>(tensor2.data<T>()),
      out_data,
      common::vectorize<int>(x.dims()),
      common::vectorize<int>(tensor2.dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

  return;
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(adaptive_layernorm_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::AdaptiveLayerNormXPUKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}