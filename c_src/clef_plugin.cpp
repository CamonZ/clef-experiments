// GatedDeltaNet recurrence adapted from Apple's MIT-licensed mlx-lm
// mlx_lm/models/gated_delta.py (v0.31.3). Copyright © 2026 Apple Inc.
// Native EMLX plugin: no Python interpreter or Python dependencies.
#include "emlx/plugin/abi.hpp"
#include "mlx/fast.h"
#include "mlx/compile.h"
#include <stdexcept>

namespace {
using namespace mlx::core;

const char* kSource = R"METAL(
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += hv_idx * Dv + b_idx * T * Hv * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) state[i] = 0.0f;
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;
    for (int t = 0; t < T; ++t) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);
        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) y[dv_idx] = static_cast<InT>(out);
        q_ += Hk * Dk;
        k_ += Hk * Dk;
        v_ += Hv * Dv;
        y += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }
)METAL";

std::optional<std::string> delta(const emlx::plugin::call_t& call, std::vector<array>& outputs) {
  try {
    const auto &q = call.operands[0], &k = call.operands[1], &v = call.operands[2];
    const auto &g = call.operands[3], &beta = call.operands[4];
    if (q.ndim() != 4 || k.shape() != q.shape() || v.ndim() != 4 || g.ndim() != 3 || beta.shape() != g.shape())
      return "invalid gated-delta input ranks/shapes";
    int B = q.shape(0), T = q.shape(1), Hk = q.shape(2), Dk = q.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);
    if (B <= 0 || T <= 0 || Hk <= 0 || Hv <= 0 || Dk <= 0 || Dv <= 0 || Dk % 32 || Hv % Hk || Dv % 4 || v.shape(0) != B || v.shape(1) != T || g.shape() != Shape{B,T,Hv})
      return "unsupported gated-delta dimensions";
    if (q.dtype() != k.dtype() || q.dtype() != v.dtype() || g.dtype() != float32 || beta.dtype() != q.dtype())
      return "invalid gated-delta dtypes";
    static const auto* kernel = new fast::CustomKernelFunction(fast::metal_kernel(
        "clef_gated_delta", {"q", "k", "v", "g", "beta", "T"}, {"y"}, kSource));
    outputs = (*kernel)({q,k,v,g,beta,array(T,int32)}, {{B,T,Hv,Dv}}, {q.dtype()},
      {32,Dv,B*Hv}, {32,4,1}, {{"InT",q.dtype()},{"Dk",Dk},{"Dv",Dv},{"Hk",Hk},{"Hv",Hv}},
      std::nullopt, false, call.stream);
    return std::nullopt;
  } catch (const std::exception& e) { return e.what(); }
}

// Match mlx-lm's compiled pointwise graphs: eager BF16 intermediates round
// differently, which matters when KnowLine reads closely spaced label logits.
std::optional<std::string> compiled_silu(const emlx::plugin::call_t& call, std::vector<array>& outputs) {
  try {
    static auto fn = compile([](const std::vector<array>& x) {
      return std::vector<array>{multiply(x[0], sigmoid(x[0]))};
    }, true);
    outputs = fn(call.operands);
    return std::nullopt;
  } catch (const std::exception& e) { return e.what(); }
}
std::optional<std::string> compiled_swiglu(const emlx::plugin::call_t& call, std::vector<array>& outputs) {
  try {
    static auto fn = compile([](const std::vector<array>& x) {
      return std::vector<array>{multiply(multiply(x[0], sigmoid(x[0])), x[1])};
    }, true);
    outputs = fn(call.operands);
    return std::nullopt;
  } catch (const std::exception& e) { return e.what(); }
}
std::optional<std::string> compiled_decay(const emlx::plugin::call_t& call, std::vector<array>& outputs) {
  try {
    static auto fn = compile([](const std::vector<array>& x) {
      auto softplus = logaddexp(add(x[1], x[2]), array(0.0f, x[1].dtype()));
      return std::vector<array>{exp(multiply(negative(exp(astype(x[0], float32))), softplus))};
    }, true);
    outputs = fn(call.operands);
    return std::nullopt;
  } catch (const std::exception& e) { return e.what(); }
}

struct Metadata {
  emlx::plugin::callback_descriptor_t callbacks[4];
  emlx::plugin::descriptor_t descriptor;
  emlx::plugin::bootstrap_v1_t bootstrap;
  Metadata() : callbacks{
      {"gated_delta",1,1,5,nullptr,1,nullptr,{Device::gpu},delta},
      {"silu",1,1,1,nullptr,1,nullptr,{Device::gpu},compiled_silu},
      {"swiglu",1,1,2,nullptr,1,nullptr,{Device::gpu},compiled_swiglu},
      {"decay",1,1,3,nullptr,1,nullptr,{Device::gpu},compiled_decay}},
    descriptor{"clef",sizeof(callbacks[0]),4,callbacks},
    bootstrap{emlx::plugin::magic_v1,sizeof(bootstrap),emlx::plugin::abi_v1,sizeof(descriptor),&descriptor} {}
};
}

extern "C" EMLX_PLUGIN_EXPORT const emlx::plugin::bootstrap_v1_t* emlx_plugin_descriptor_v1() noexcept {
  static const auto* metadata = new Metadata();
  return &metadata->bootstrap;
}
