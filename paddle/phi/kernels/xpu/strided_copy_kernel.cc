/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/strided_copy_kernel.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = common::make_ddim(out_stride);
  meta.dims = common::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  PADDLE_ENFORCE_EQ(input.dims(),
                    out->dims(),
                    common::errors::InvalidArgument(
                        "Input shape(%s) must be equal with out shape(%s).",
                        input.dims(),
                        out->dims()));

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    common::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  // use XPUCopyTypeTrait to deal with double and int16_t copy instead of
  // XPUTypeTrait
  using XPUType = typename XPUCopyTypeTrait<T>::Type;

  int r = 0;
  auto input_data = reinterpret_cast<const XPUType*>(input.data<T>());
  auto output_data = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));
  PADDLE_ENFORCE_NOT_NULL(output_data,
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));
  if (input.numel() == 1) {
    r = xpu::copy<XPUType>(dev_ctx.x_context(), input_data, output_data, 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    r = xpu::strided_copy<XPUType>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   common::vectorize<int64_t>(input.dims()),
                                   common::vectorize<int64_t>(out->dims()),
                                   common::vectorize<int64_t>(input.strides()),
                                   common::vectorize<int64_t>(out->strides()));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_copy");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_copy,
                   XPU,
                   ALL_LAYOUT,
                   phi::StridedCopyKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16) {}
