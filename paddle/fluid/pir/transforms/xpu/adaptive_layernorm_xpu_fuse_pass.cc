// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/xpu/adaptive_layernorm_xpu_fuse_pass.h"
#include <optional>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {
/*
fuse malmul + act to fc_xpu
For example:
graph:

                  x   w   b         scale_in
                  |   |   |           | 
                  ---------           | 
                      |               |  
                      |               |  factor
                      |               |  /
                    layer_norm      scale
                      |               |
                      |               |               
                      |               |               
                      |               |                
                      -----------------             add_in
                             |                         |
                             |                         |
                          multiply                     |
                             |                         |
                             ---------------------------
                                         |
                                        add
                                         |
                                        output
------------------------------------------------------
After the pass is applied:
               x   w  b  scale_in  add_in
               |   |  |   |         |
               ----------------------
                      |
            adaptive_layernorm_xpu
                      |
                      |
                    Output
*/

class AdaptiveLayernormPattern : public paddle::drr::DrrPatternBase {
 private:
  bool transpose_w_;

 public:
  std::string name() const override { return "AdaptiveLayernormPattern"; }
  uint32_t benefit() const override { return 3; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});

    const auto &full = pat.Op(paddle::dialect::FullOp::name(),
                              {{"shape", pat.Attr("full_shape")},
                               {"value", pat.Attr("full_value")},
                               {"dtype", pat.Attr("full_dtype")},
                               {"place", pat.Attr("full_place")}});
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("scale_bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});

    const auto &multiply = pat.Op(paddle::dialect::MultiplyOp::name());

    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    layernorm({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("b")},
              {&pat.Tensor("layer_norm_out"),
               &pat.Tensor("mean_out_0"),
               &pat.Tensor("variance_out_0")});
    scale({&pat.Tensor("scale_in"), &full()},
          {&pat.Tensor("scale_out")});
    multiply({&pat.Tensor("layer_norm_out"), &pat.Tensor("scale_out")},
             {&pat.Tensor("multiply_out")});
    add({&pat.Tensor("multiply_out"), &pat.Tensor("add_in")},
        {&pat.Tensor("output")});

    // Constraints
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto scale_in_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("scale_in"));
      auto add_in_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("add_in"));

      if ((x_shape.size() == scale_in_shape.size()) &&
          (scale_in_shape.size() == add_in_shape.size())) {
        return true;
      }

      return true;
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_value");
        });

    const auto &adaptive_layernorm_xpu =
        res.Op(paddle::dialect::AdaptiveLayernormXpuOp::name(),
               {{
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"factor", fused_factor},
                   {"scale_bias", pat.Attr("scale_bias")},
                   {"bias_after_scale", pat.Attr("bias_after_scale")},
               }});
    adaptive_layernorm_xpu(
        {
            &res.Tensor("x"),
            &res.Tensor("w"),
            &res.Tensor("b"),
            &res.Tensor("scale_in"),
            &res.Tensor("add_in"),
        },
        {&res.Tensor("output")});
  }
};

class AdaptiveLayernormXpuFusePass : public pir::PatternRewritePass {
 public:
  AdaptiveLayernormXpuFusePass()
      : pir::PatternRewritePass("adaptive_layernorm_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<AdaptiveLayernormPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateAdaptiveLayernormXpuFusePass() {
  return std::make_unique<AdaptiveLayernormXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(adaptive_layernorm_xpu_fuse_pass,
                 AdaptiveLayernormXpuFusePass);
