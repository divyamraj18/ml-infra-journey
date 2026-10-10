# ML Infrastructure & Inference Engineering: Master Blueprint & Curriculum

> **Handoff / State Tracker** (Update this as you progress)
> - **Student:** Divyam Raj (Senior SRE )
> - **Current Module:** Module 1 (GPU Architecture & Low-Level CUDA Programming)
> - **Current Submodule:** Section 1.1 - GPU Hardware Mechanics & CUDA Thread Hierarchy
> - **Active Task:** Assignment 1.1 (Vector Addition in CUDA C++)
> - **Status:** In Progress / Not Started

---

## Overview & Flow
This repository documents an end-to-end transition from SRE to ML Infrastructure & Inference Engineering, covering low-level CUDA optimization, production inference engines (vLLM/Triton), distributed systems (Ray/NCCL), cloud-native orchestration, and SRE/observability.

---

## MODULE 1: GPU Architecture & Low-Level CUDA Programming
*Directory: `01-cuda-fundamentals/`*

### Submodule 1.1: GPU Hardware & Execution Model
- **Core Topics:** Streaming Multiprocessors (SMs), CUDA Cores vs. Tensor Cores, Registers, SRAM (Shared Memory) vs. HBM (Global Memory), SIMT Architecture, Warps, Warp Divergence, and Thread Index Mapping.
- **Assignment 1.1:** Write `vector_add.cu` with `CUDA_CHECK` error macros, host-to-device memory allocation/copies (`cudaMalloc`/`cudaMemcpy`), and execution time measurement using `cudaEvent_t`.
- **Deliverables:** `01-cuda-fundamentals/assignments/01-vector-add/`

### Submodule 1.2: Memory Coalescing, Shared Memory, and Tiling
- **Core Topics:** Global memory coalescing rules, bank conflicts in shared memory, and tiled matrix multiplication algorithms.
- **Assignment 1.2:** Implement Naive Matrix Multiplication vs. Shared Memory Tiled Matrix Multiplication ($4096 \times 4096$).
- **Deliverables:** `01-cuda-fundamentals/assignments/02-matrix-mul-tiled/`

### Submodule 1.3: Profiling & Kernel Optimization
- **Core Topics:** Compute-bound vs. Memory-bound kernels, NVIDIA Nsight Systems (`nsys`), and Nsight Compute (`ncu`).
- **Assignment 1.3:** Profile the matrix multiplication kernel, analyze warp occupancy, stall reasons, and optimize memory throughput.
- **Deliverables:** `01-cuda-fundamentals/kernels/` & `docs/`

### Module 1 Mini-Project: Custom CUDA FlashAttention-Style Softmax
- **Goal:** Write a fused Online Softmax / Attention reduction kernel in CUDA C++, wrap it with Python via **PyBind11**, and benchmark against PyTorch `torch.softmax`.
- **Deliverables:** `01-cuda-fundamentals/kernels/custom_softmax/`

---

## MODULE 2: High-Performance Inference Engines & Optimizations
*Directory: `02-inference-engines/`*

### Submodule 2.1: LLM Inference Mechanics & Bottlenecks
- **Core Topics:** Compute-bound (Prefill) vs. Memory-bound (Decode) phases, KV-Cache memory footprint, PagedAttention architecture (vLLM), Continuous Batching, and Speculative Decoding.
- **Assignment 2.1:** Build a synthetic Python benchmarking script measuring Time-To-First-Token (TTFT) and Inter-Token Latency (ITL) under varying concurrency.

### Submodule 2.2: Advanced Serving & Quantization
- **Core Topics:** Triton Inference Server architecture, Dynamic Batching, Model Ensembles, Shared Memory IPC, Post-Training Quantization (PTQ) vs QAT (AWQ, GPTQ, FP8).
- **Assignment 2.2:** Deploy a custom Python/C++ backend on Triton Inference Server with Dynamic Batching enabled.

### Module 2 Mini-Project: vLLM vs. Triton Benchmarking Suite
- **Goal:** End-to-end benchmark suite comparing throughput and latency of FP16 vs. Quantized (AWQ/FP8) models under peak load.
- **Deliverables:** `02-inference-engines/vllm-benchmarks/` & `triton-custom-backend/`

---

## MODULE 3: Distributed Systems & Compute Orchestration
*Directory: `03-distributed-systems/`*

### Submodule 3.1: Inter-GPU Communication & Parallelism
- **Core Topics:** Tensor Parallelism (Megatron-LM style), Pipeline Parallelism, Sequence Parallelism, NCCL internals, Ring All-Reduce, NVLink, and RDMA/RoCE mechanics.
- **Assignment 3.1:** Write a multi-GPU PyTorch distributed script using NCCL primitives to perform manual Tensor Parallel MatMul across 2 GPUs.

### Submodule 3.2: Cluster Scheduling & Orchestration
- **Core Topics:** Ray architecture (Ray Core, Ray Serve), KubeRay operators, Slurm scheduler vs. Kubernetes for AI workloads.
- **Assignment 3.2:** Build a Ray Serve DAG with dynamic request routing and horizontal auto-scaling.

### Module 3 Mini-Project: Fault-Tolerant Distributed Ray Serve on K8s
- **Goal:** Deploy a multi-node, fault-tolerant Ray Serve cluster using KubeRay on a Kubernetes cluster serving a sharded LLM.
- **Deliverables:** `03-distributed-systems/kuberay-manifests/` & `ray-serve-dag/`

---

## MODULE 4: Cloud-Native ML Infrastructure & MLOps
*Directory: `04-cloud-native-mlops/`*

### Submodule 4.1: Kubernetes GPU Infrastructure & Storage
- **Core Topics:** NVIDIA GPU Operator, Device Plugins, MIG (Multi-Instance GPU), High-throughput object storage (MinIO/Ceph), GPUDirect Storage (GDS).
- **Assignment 4.1:** Configure a Kubernetes cluster node with NVIDIA GPU Operator and allocate MIG slices to specific pods.

### Submodule 4.2: Control Planes & CI/CD for ML
- **Core Topics:** Model Registries, KServe / Seldon Core, Zero-downtime model weight updates and canary deployments.
- **Assignment 4.2:** Setup automated canary deployment for model weights without dropping active client connections.

### Module 4 Mini-Project: Infrastructure-as-Code Deployment
- **Goal:** Use Terraform and Helm to provision a complete GPU-enabled Kubernetes cluster with MinIO weight storage and automated KServe serving.
- **Deliverables:** `04-cloud-native-mlops/terraform/` & `helm-kserve/`

---

## MODULE 5: Observability, SRE & Performance Profiling
*Directory: `05-observability-profiling/`*

### Submodule 5.1: GPU Metrics, Tracing & SLIs/SLOs
- **Core Topics:** DCGM (Data Center GPU Manager) metrics export, OpenTelemetry (OTel) distributed tracing across API Gateway -> Router -> vLLM worker, and defining AI SLIs/SLOs (TTFT, ITL).
- **Assignment 5.1:** Construct a Prometheus + Grafana dashboard using DCGM Exporter and vLLM OTel metrics to detect memory fragmentation and GPU throttling.
- **Deliverables:** `05-observability-profiling/dcgm-prometheus/` & `otel-tracing/`

---

## CAPSTONE PROJECTS
*Directory: `capstone-projects/`*

### Capstone 1: Enterprise-Grade LLM Gateway & Distributed Serving Platform
- **Goal:** Build a production-grade LLM gateway featuring continuous batching, dynamic model loading/unloading, KV-cache eviction policies, Prometheus observability, and Ray autoscaling on Kubernetes.
- **Deliverables:** `capstone-projects/01-llm-gateway-platform/`

### Capstone 2: High-Throughput Custom Multi-Modal Inference Engine
- **Goal:** Build a multi-stage pipeline (Vision Model + LLM) deployed on Triton Inference Server using dynamic batching, shared memory IPC, and custom C++/CUDA preprocessing kernels.
- **Deliverables:** `capstone-projects/02-multimodal-triton-pipeline/`