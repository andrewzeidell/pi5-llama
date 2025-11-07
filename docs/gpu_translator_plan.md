# GPU Translator Roadmap — Raspberry Pi 5 LLM Runtime

## Overview

This document outlines the technical roadmap for transforming the current **Vulkan compute prototype** into a **lightweight GPU‑accelerated LLM runtime** on Raspberry Pi 5.  
The goal is a hybrid compute system capable of running smaller models (TinyLlama, Phi‑1.5, Mistral‑7B‑quantized) with efficient GPU offload for matrix and attention workloads, while balancing host memory and GPU constraints on the Pi 5.

**Vision:**  
Create a reusable inference library — `pi5_llm_rt` — integrating a minimal Vulkan scheduler with model parsing, lazy tensor memory management, and mixed‑CPU/GPU dispatch, optimized for the VideoCore VII GPU.

---

## Current Capabilities Summary

| Component | Description | Files |
|------------|--------------|-------|
| **Compute Backend (Vulkan)** | Supports matrix multiplication (`matmul`), softmax (`softmax_rows`), and fused attention (`attn_fused`) | [`src/vk.rs`](../src/vk.rs), [`shaders/matmul_f16_t16x16.comp`](../shaders/matmul_f16_t16x16.comp), [`shaders/softmax_rows.comp`](../shaders/softmax_rows.comp), [`shaders/attn_fused.comp`](../shaders/attn_fused.comp) |
| **SPIR‑V shaders** | Precompiled variants (`.spv`) for both f16 and f32 workloads optimized for Pi VideoCore VII | `shaders/*.spv` |
| **CPU Backend** | Provides reference operations for correctness and baseline performance | [`src/cpu.rs`](../src/cpu.rs) |
| **Backend Abstraction** | Unified interface for switching between Vulkan and CPU computations | [`src/backend.rs`](../src/backend.rs) |

---

## Phase 1 — GPU Operator Expansion

**Objective:** Extend the Vulkan backend to cover essential transformer sub‑operators for larger model graphs.

### Planned GPU Kernels

| Operator | Action | Deliverable |
|-----------|---------|-------------|
| **LayerNorm** | Develop shader using per‑row mean/variance reduction and normalization | `shaders/layernorm.comp` + `layernorm.spv` |
| **Activation (SiLU, GELU)** | FP16‑optimized pointwise ops for feedforward residuals | `shaders/activation_silu.comp` / `shaders/activation_gelu.comp` |
| **RoPE (Rotary Positional Encoding)** | Efficient elementwise complex mix kernel | `shaders/rope_rotary.comp` |
| **Linear Projection** | Fused matmul + bias add | `shaders/projection_bias.comp` |
| **Residual Add** | Simple buffer addition for skip connections | `shaders/add_f16.comp` |

### Supporting Rust Modules
- [`src/ops/layernorm.rs`] – dispatch and host buffer setup.
- [`src/ops/activation.rs`] – generic unary kernels with f16/f32 paths.
- [`src/ops/rope.rs`] – RoPE angle precomputation and dispatch.

### Dependencies / Research
- Confirm **VK_KHR_shader_float16_int8** and FP16 storage buffer support in Pi 5 Mesa V3D driver.
- Benchmark local memory availability (≤64 KB per workgroup).
- Validate subgroup ops performance for reduction patterns.

---

## Phase 2 — Model Graph Runtime

**Objective:** Implement an efficient runtime scheduler to execute transformer layers with **selective CPU/GPU dispatch**.

### Key Features
- **Computation graph definition:** define model sub‑graphs in Rust (`src/graph/`).
- **Execution planner:** assign ops to GPU or CPU based on cost heuristics.
- **Tensor allocator:** device‑aware buffer pool (persistent VkBuffers + pinned CPU memory).
- **Stream executor:** queue‑sync system to overlap upload/download with compute.

### Deliverables
- [`src/runtime/graph.rs`] – DAG construction, node typing.
- [`src/runtime/scheduler.rs`] – CPU↔GPU dispatch manager.
- [`src/runtime/tensor.rs`] – device‑agnostic tensor handle.

### Notes
The scheduler will initially target static compute graphs (Transformer block unrolled). Later phases may add dynamic shapes for token streaming.

---

## Phase 3 — Integration with Lightweight LLM Formats

**Objective:** Enable model import from standard lightweight storage formats.

### Supported Imports
| Format | Parser Module | Description |
|---------|----------------|-------------|
| **GGUF / GGML** | `src/io/gguf_parser.rs` | Load tensors and metadata from TinyLlama quantized formats. |
| **ONNX (subset)** | `src/io/onnx_loader.rs` | Parse small transformer models for interoperability. |

### Deliverables
- GGUF parameter reader with layer binding (`gguf_parser.rs`)
- Optional ONNX fallback for academic models
- Converter scripts under `tools/convert_gguf_to_vk.py`

### Dependencies / Research
- GGUF schema version 3 reference.
- Toy model checkpoints for Phi‑1.5 or TinyLlama.
- Weight quantization interoperability (Q4_K / FP16 hybrid).

---

## Phase 4 — Optimization and Testing

**Objective:** Ensure runtime correctness, numerical stability, and real performance gains.

### Focus Areas
- **Correctness:** Verify GPU ops versus CPU reference (`tests/test_ops.rs`).
- **Performance:** Benchmark attention throughput and memory bandwidth.
- **Profiling:** Integrate micro‑timers around Vulkan dispatches.
- **FP16 Emulation Fallback:** Graceful degradation on FP32 path devices.

### Deliverables
- [`benchmarks/bench_gpu_ops.rs`] – Criterion-based timing suite.
- [`tests/test_graph_accuracy.rs`] – Layer‑by‑layer comparison.
- JSON summary export for integration with CI dashboards.

### Research Needs
- Optimal workgroup sizes for VideoCore VII.
- Kernel fusion (e.g., matmul+softmax+projection fused kernel).

---

## Phase 5 — Packaging & Deployment

**Objective:** Make the runtime distributable as a lightweight inference library for Raspberry Pi deployments.

### Deliverables
| Artifact | Description |
|-----------|-------------|
| `crate: pi5_llm_rt` | Public Rust crate exposing `VkRuntime` + graph interface |
| `examples/pi5_llama.rs` | Minimal example running TinyLlama inference |
| `docs/usage_gpu_runtime.md` | Developer documentation for integration |
| `pi5_llm_rt.a` / `libpi5_llm_rt.so` | Prebuilt static and shared libs |
| `pip wheel (optional)` | Python FFI wrapper with `pyO3` |

### Additional Goals
- **Cross‑compile toolchain:** Ensure ARM64 Vulkan linkage with `--target=aarch64-unknown-linux-gnu`.
- **Packaging scripts:** CMake + Cargo hybrid for mixed builds.
- **MkDocs Integration:** Include this file in `mkdocs.yml` under *Development → GPU Translator Roadmap*.

---

## LLM Integration on Raspberry Pi 5

### 🎯 Goal
Establish full integration between the **Vulkan translator runtime** and **local lightweight LLMs** (e.g., `ollama`, `llama.cpp`, or custom Rust/Python wrappers).

---

### 1. System Integration Overview

The Vulkan runtime (`pi5_llm_rt`) will expose its compute graph and tensor API as an **FFI‑friendly C interface**:

```
libpi5_llm_rt/
└── include/
    └── pi5_llm_rt.h     # C header exposing runtime entry points
```

**API Core:**
```c
// Simplified C-header example
typedef struct VkRuntime VkRuntime;

VkRuntime* pi5_create_runtime();
void pi5_destroy_runtime(VkRuntime* rt);
int pi5_load_model(VkRuntime* rt, const char* path);
int pi5_infer(VkRuntime* rt, const float* input, float* output);
```

These symbols will be compatible with:
- `llama.cpp`: via shared library binding (`dlopen("libpi5_llm_rt.so")`)
- `ollama`: via modular GPU plugin backend concept
- Custom Python clients via `ctypes` or `pyO3` bindings

**Deliverable:**  
🧩 *Expose `libpi5_llm_rt.so` ABI header for external runtimes.*

---

### 2. Inference Data Flow

```mermaid
flowchart LR
A[Tokens (LLM prompt)] --> B[Tokenizer → Embedding (CPU)]
B --> C[GPU VKCompute Graph]
C --> D[Attention/MLP layers]
D --> E[Logits → Decoder (CPU/Python host)]
E --> F[Next token sampling/output]
```

- **Tokenization:** Performed on host using Rust or Python LLM tooling.  
- **Embedding matrix fetch:** Copied to GPU buffer once per batch.  
- **Compute path:**  
  - F16 matrix multiply via `matmul_f16_t16x16.comp`  
  - Softmax and fused attention via `attn_fused.comp`  
  - Layernorm and activation dispatched via Vulkan scheduler.  
- **Output:** GPU produces logits → CPU sampling loop.

---

### 3. Installation & Setup Instructions

**Prerequisites**
```bash
sudo apt update
sudo apt install mesa-vulkan-drivers vulkan-tools build-essential cargo
```

**Driver Expectation:**  
Raspberry Pi 5 ships with Mesa’s **V3DV Vulkan** driver supporting FP16 shader ops.  
Validate capabilities:
```bash
vulkaninfo | grep V3DV
```

**Build from source**
```bash
git clone https://github.com/yourorg/pi5-llama.git
cd pi5-llama
cargo build --release --target=aarch64-unknown-linux-gnu
```

**Optional Python Binding**
```bash
maturin build --release
pip install ./target/wheels/pi5_llm_rt*.whl
```

---

### 4. Runtime Invocation Examples

**CLI Example**
```bash
./target/release/pi5_llm_rt --model tinyllama.gguf --prompt "Hello Pi"
```

**Rust snippet**
```rust
use pi5_llm_rt::VkRuntime;

let mut rt = VkRuntime::new();
rt.load_model("models/tinyllama.gguf").unwrap();
let logits = rt.infer("Hello Pi").unwrap();
println!("{:?}", logits);
```

**Python (via pyO3)**
```python
import pi5_llm_rt
rt = pi5_llm_rt.Runtime()
rt.load_model("tinyllama.gguf")
print(rt.infer("Hello Pi"))
```

---

### 5. Constraints & Optimization Guidelines

> ⚙️ **Hardware Limits:**  
> - Pi 5 VRAM shared memory: ~768 MB usable for GPU buffers.  
> - Use FP16 exclusively to fit model weights.  
> - Prefer static buffer pools over per‑token reallocation.

**Performance Tips**
- Tune *workgroup sizes* between 8–16×16 for attention kernels.
- Adjust *batch size* to fit available RAM (≤ 64 tokens recommended).
- Use quantized models (Q4_K, GGUF) when memory‑bound.

---

### 6. Future Extension Hooks

| Extension | Integration Path |
|------------|------------------|
| **Quantization (Q3/Q4/Q5)** | Extend GGUF loader to support per‑layer dequantization kernels |
| **LoRA adapters** | Inject additional linear ops into existing compute graph nodes |
| **KV cache sharing** | Memory‑mapped buffers across sessions for faster generation |
| **Profiler hooks** | Expose timing APIs for frontends (llama.cpp, Ollama) |
| **Dynamic offload policy** | CPU/GPU split tuned in runtime config |

**Implementation Note:**  
🔧 *These hooks will be defined as optional Vulkan “extensions” discoverable at runtime through the FFI.*

---

## Research Outlook

| Topic | Notes |
|-------|-------|
| **FP16 Precision** | Evaluate possible compute error accumulation on matrix ops. |
| **Memory Bandwidth** | Analyze upload/download overhead and attempt buffer reuse. |
| **Dynamic Quantization** | Explore FP16→INT8 per‑layer conversion to fit larger models. |
| **Runtime Offloading Policy** | Evaluate heuristics for deciding CPU vs GPU execution per operator. |

---

## Summary Milestones

| Phase | Target Outcome |
|--------|----------------|
| Phase 1 | Complete core set of transformer GPU kernels |
| Phase 2 | Establish execution runtime & selective scheduler |
| Phase 3 | Support GGUF and ONNX imports |
| Phase 4 | Validation + benchmark suite |
| Phase 5 | Library packaging & deployment |

---

*This plan will guide incremental implementation and validation efforts to turn the Vulkan compute foundations into a fully operational Raspberry Pi 5 GPU‑accelerated LLM runtime.*

---

## Benchmarking and Remote Testing Strategy

### 🎯 Goal
Enable rapid validation of GPU/CPU inference performance and ensure functional parity between the host development system and Raspberry Pi 5 target hardware.

---

### 1. Host–Target Testing Structure

- **Cross‑compile** Rust benchmarks for ARM64 on host:
  ```bash
  cargo build --release --target=aarch64-unknown-linux-gnu --bin benchmark_runner
  ```
- **Deploy** to Pi 5 over SSH and run automatically via script:
  ```bash
  ./scripts/run_benchmarks_pi.sh <hostname> <user>
  ```
- Script workflow:
  1. Build on host using cross‑compiler
  2. SCP deploy to Raspberry Pi 5 (`~/benchmarks/`)
  3. Remote execute via SSH, collect logs under `results/`
  4. Optionally upload logs back to host for comparison

---

### 2. Benchmark Types

Type | Example Ops | Description |
|------|--------------|-------------|
**Unit Benchmarks** | `matmul`, `softmax_rows`, `attention_fused` | Measure single‑kernel dispatch latency with random tensors |
**Integration Benchmarks** | TinyLlama, synthetic input graphs | Validate model‑level throughput (mini‑runs) |
**System Health** | Vulkan driver, memory throughput | Confirm driver stability, GPU availability, and sustained clock |

---

### 3. Execution Scripts

Example automation for repeatable Pi runs:
```bash
#!/bin/bash
# Usage: ./scripts/run_benchmarks_pi.sh <hostname> <user>

HOST=$1
USER=$2
TARGET_DIR=/home/$USER/benchmarks
BIN=target/aarch64-unknown-linux-gnu/release/benchmark_runner

# Build & copy to Pi
cargo build --release --target=aarch64-unknown-linux-gnu
scp $BIN $USER@$HOST:$TARGET_DIR/

# Run remotely and collect results
ssh $USER@$HOST "cd $TARGET_DIR && ./benchmark_runner --all > benchmark_log.txt"
scp $USER@$HOST:$TARGET_DIR/benchmark_log.txt results/
```

---

### 4. Key Metrics Captured

- ⏱ **Execution time (ms)** per kernel
- 📈 **Tensor size scaling** vs timing
- ⚙️ **Throughput (GFLOPS)** CPU vs GPU
- 🌡 **Thermal data:** collect from `/sys/class/thermal/thermal_zone0/temp`
- 📊 **Summary reporting:** convert logs → Markdown or JSON

---

### 5. CI Integration

- Integrate benchmark script invocation into CI pipeline:
  - Host triggers benchmark runner over SSH on the Pi 5 test node.
  - Results automatically uploaded to `results/benchmark_log.md`.
  - Markdown summaries appended for regression visualization.
- **Optional Extensions:**
  - Use `criterion.rs` metrics directly in benchmark builds.
  - Compare previous runs for performance deltas and flag anomalies.

---

### 6. Deliverables

File | Purpose |
|------|----------|
[`benchmarks/benchmark_runner.rs`](../benchmarks/benchmark_runner.rs) | Central benchmark orchestrator integrating Criterion and timing macros |
[`scripts/run_benchmarks_pi.sh`](../scripts/run_benchmarks_pi.sh) | Host‑to‑Pi automation script |
[`results/benchmark_log.md`](../results/benchmark_log.md) | Centralized run log and Markdown summary for comparisons |

---

### 7. Action Steps for Remote Testers

1. Prepare Pi 5 system with Vulkan support:
   ```bash
   sudo apt install vulkan-tools mesa-vulkan-drivers
   vulkaninfo | grep V3DV
   ```
2. Run automated benchmark flow from host:
   ```bash
   ./scripts/run_benchmarks_pi.sh pi5.local pi
   ```
3. Review `results/benchmark_log.md` to compare GPU vs CPU metrics.

---

*This strategy ensures reproducible, automated validation of every GPU kernel and inference optimization on real Raspberry Pi 5 hardware, enabling continuous integration and performance assurance without requiring local manual deployment.*