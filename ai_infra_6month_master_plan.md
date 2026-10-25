# ML Infrastructure & Inference Engineering: Master Blueprint & Curriculum (6-Month Plan)

> **Handoff / State Tracker** (Update this as you progress)  
> 

> - **Student:** Divyam Raj (Senior Engineer Infra)  
> - **Current Module:** Week 1 (GPU Architecture & Low-Level CUDA Programming)  
> - **Current Submodule:** Week 1 — Assignment 1.1 vector_add.cu  
> - **Active Task:** Run `nvcc -arch=sm_75 vector_add.cu -o vector_add` on a Colab T4 and check kernel time plus mismatches  
> - **Status:** In Progress

---

## Overview & Flow

This repository documents an end-to-end 6-Month Master Learning Plan (26 Weeks) designed for transitioning from core infrastructure/SRE roles to a Niche AI Infrastructure & Distributed Compute Engineer. It combines low-level GPU CUDA optimization, high-performance inference engines (vLLM/Triton), distributed systems (Ray/NCCL/KubeRay), cloud-native MLOps, and SRE/observability.

### Repository layout

```text
01-cuda-fundamentals/                 # Month 1 (Weeks 1–4)
  docs/week1/                         # Days 1–7
  docs/week2/                         # Days 8–14
  assignments/01-vector-add/          # Assignment 1.1
  assignments/02-matrix-mul-tiled/    # Assignment 1.2
  kernels/                            # Assignment 1.3
  kernels/custom_softmax/             # Week 4
02-inference-engines/                 # Month 2 (Weeks 5–6)
  ttft-itl-benchmark/                 # Assignment 2.1
  vllm-benchmarks/                    # Assignment 2.2
  triton-custom-backend/              # Assignment 2.2
03-kubernetes-for-ml/                 # Month 2 (Weeks 7–8)
  gpu/
  scheduling/
04-ray/                               # Month 3 (Weeks 9–12)
  intro/
  memory/
  scheduling/
  fault-tolerance/
05-kuberay/                           # Month 4 (Weeks 13–16)
  crds/
  ray-train/
  ray-data/
  ray-serve/
06-production/                        # Month 5 (Weeks 17–20)
  cluster/
  chaos/
  profiling/
07-oss-and-interviews/                # Month 6 (Weeks 21–26)
  contributions/
  system-design/
  career/
cloud-native-mlops/                   # Integrated add-on
  gpu-operator/
  storage/
  serving/
observability/                        # Integrated add-on
  dcgm-prometheus/
  otel-tracing/
capstone-projects/
  01-llm-gateway-platform/
  02-multimodal-triton-pipeline/
```

---

## Phase 1: Distributed Systems, AI Hardware Fundamentals & CUDA Optimization (Month 1 & 2\)

### Month 1: Hardware, Interconnects, GPU Mechanics & Low-Level CUDA

#### Week 1: GPU Architecture & Memory (Daily 2-3 Hours)

* **Day 1:** NVIDIA GPU Architecture Overview (Streaming Multiprocessors, Tensor Cores, CUDA Cores, Registers, SRAM vs. HBM, SIMT Architecture, Warps, Warp Divergence, Thread Index Mapping). Difference between CPU and GPU workloads.  
* **Day 2:** GPU Memory Hierarchy: HBM (High Bandwidth Memory) vs. SRAM vs. DRAM. Understanding Memory Bandwidth bottlenecks.  
* **Day 3:** GPU Out-of-Memory (OOM) Errors: Understanding why LLM training runs out of memory (Activation memory, Optimizer states, Model weights).  
* **Day 4:** Introduction to CUDA execution model: Grids, Blocks, Threads, and Coalesced Execution.  
* **Day 5:** Memory Optimization techniques: Mixed Precision Training (FP16, BF16, FP8) and Gradient Checkpointing.  
* **Day 6-7 (Weekend):** Review NVIDIA Whitepapers (Ampere/Hopper/Blackwell architecture) \+ Practical: Profile a simple PyTorch script using PyTorch Profiler to see GPU utilization.  
* **Assignment 1.1:** Write `vector_add.cu` with `CUDA_CHECK` error macros, host-to-device memory allocation/copies (`cudaMalloc`/`cudaMemcpy`), and execution time measurement using `cudaEvent_t`. (Deliverable: `01-cuda-fundamentals/assignments/01-vector-add/`)

#### Week 2: Memory Coalescing, Shared Memory, Tiling & Cluster Interconnects

* **Day 8:** Intra-node communication: NVIDIA NVLink and NVSwitch. How GPUs talk inside a single box (e.g., DGX H100). Global memory coalescing rules, bank conflicts in shared memory, and tiled matrix multiplication algorithms.  
* **Day 9:** Inter-node communication: InfiniBand (IB) vs. RoCE (RDMA over Converged Ethernet). Why TCP/IP is too slow for LLM training.  
* **Day 10:** Deep dive into RDMA (Remote Direct Memory Access) \- bypassing the CPU kernel for ultra-low latency.  
* **Day 11:** NCCL (NVIDIA Collective Communications Library): Concepts of AllReduce, AllGather, ReduceScatter, and Broadcast.  
* **Day 12:** Debugging Networking Bottlenecks: NCCL environment variables (`NCCL_DEBUG=INFO`) and network topology awareness.  
* **Day 13-14 (Weekend):** Study NCCL topologies and draw a diagram of how 8 GPUs across 2 nodes communicate during an AllReduce step.  
* **Assignment 1.2:** Implement Naive Matrix Multiplication vs. Shared Memory Tiled Matrix Multiplication (\$4096 \\times 4096\$). (Deliverable: `01-cuda-fundamentals/assignments/02-matrix-mul-tiled/`)

#### Week 3: Distributed Training Mechanics (Part 1\) & Kernel Profiling

* **Day 15:** Data Parallelism (DP): How it works, Synchronous vs. Asynchronous SGD, Parameter Server vs. AllReduce. Compute-bound vs. Memory-bound kernels, NVIDIA Nsight Systems (`nsys`), and Nsight Compute (`ncu`).  
* **Day 16:** Distributed Data Parallel (DDP) in PyTorch. Why DDP scales better than standard DP.  
* **Day 17:** Hands-on: Write a simple PyTorch DDP script and run it locally simulating multiple processes.  
* **Day 18:** Pipeline Parallelism (PP): Splitting model layers across multiple GPUs. Interleaved 1F1B (One Forward, One Backward) scheduling.  
* **Day 19:** Bubble problem in Pipeline Parallelism and how to minimize it.  
* **Day 20-21 (Weekend):** Implement or trace a small pipeline-parallel architecture using toy layers to understand communication wait times.  
* **Assignment 1.3:** Profile the matrix multiplication kernel, analyze warp occupancy, stall reasons, and optimize memory throughput. (Deliverable: `01-cuda-fundamentals/kernels/` & `docs/`)

#### Week 4: Distributed Training Mechanics (Part 2\) & Custom FlashAttention

* **Day 22:** Tensor Parallelism (TP): Splitting a single Layer/Matrix multiplication across multiple GPUs (Megatron-LM style). Custom CUDA FlashAttention-Style Softmax: Write a fused Online Softmax / Attention reduction kernel in CUDA C++, wrap it with Python via PyBind11, and benchmark against PyTorch `torch.softmax`.  
* **Day 23:** Understanding Column Parallel Linear and Row Parallel Linear layers. NCCL communication overhead in TP.  
* **Day 24:** ZeRO (Zero Redundancy Optimizer) by DeepSpeed: ZeRO-1 (Optimizer state partitioning), ZeRO-2 (Gradient partitioning), ZeRO-3 (Parameter partitioning).  
* **Day 25:** CPU/NVMe Offloading in DeepSpeed/ZeRO-Infinity for ultra-large models.  
* **Day 26-27 (Weekend):** Compare TP, PP, and ZeRO. Write a conceptual summary comparing their communication frequency and bandwidth demands.  
* **Deliverable:** `01-cuda-fundamentals/kernels/custom_softmax/`

### Month 2: Production Storage, Checkpointing, Inference Mechanics & Quantization

#### Week 5: Storage Architectures & LLM Inference Bottlenecks

* **Day 28:** Storage Bottlenecks: Training data ingestion vs. Checkpoint saving. Compute-bound (Prefill) vs. Memory-bound (Decode) phases, KV-Cache memory footprint, PagedAttention architecture (vLLM), Continuous Batching, and Speculative Decoding.  
* **Day 29:** Parallel File Systems (PFS): Lustre, WekaIO, GPFS (IBM Spectrum Scale). Why standard NFS/S3 fails under heavy checkpointing.  
* **Day 30:** Data caching layers (Alluxio, Ray Data caching mechanics).  
* **Day 31:** High-throughput data loading: PyTorch DataLoader bottlenecks, WebDataset format for streaming massive image/text datasets.  
* **Day 32:** Checkpoint serialization/deserialization times and its impact on cluster uptime.  
* **Day 33-34 (Weekend):** Architect a high-level storage design for a cluster training a 100B parameter model, optimizing for both streaming data and frequent checkpoints.  
* **Assignment 2.1:** Build a synthetic Python benchmarking script measuring Time-To-First-Token (TTFT) and Inter-Token Latency (ITL) under varying concurrency. (Deliverable: `02-inference-engines/ttft-itl-benchmark/`)

#### Week 6: Fault Tolerance, Advanced Serving & Quantization

* **Day 35:** Cost of failure: What happens when 1 node out of 1024 dies during a 3-month training run? Triton Inference Server architecture, Dynamic Batching, Model Ensembles, Shared Memory IPC, Post-Training Quantization (PTQ) vs QAT (AWQ, GPTQ, FP8).  
* **Day 36:** Elastic Training basics: Dynamic cluster resizing during training (TorchElastic).  
* **Day 37:** Automated health checks, node fencing, and identifying silent data corruption (e.g., bad GPU memory leading to NaN loss).  
* **Day 38:** Introduction to Slurm Workload Manager vs. Kubernetes for AI workloads.  
* **Day 39:** Pros and cons of Slurm (HPC history, bare-metal speed) vs. Kubernetes (Cloud-native, containerized ecosystem).  
* **Day 40-41 (Weekend):** Read case studies on how Meta or OpenAI design their scheduling infrastructure (Kubernetes vs. Slurm customized layers).  
* **Assignment 2.2:** Deploy a custom Python/C++ backend on Triton Inference Server with Dynamic Batching enabled. (Deliverables: `02-inference-engines/vllm-benchmarks/` & `02-inference-engines/triton-custom-backend/`)

#### Week 7 & 8: Deep Dive into Kubernetes for ML (Core Infrastructure Integration)

* **Day 42-45:** GPU Virtualization in K8s: NVIDIA Container Toolkit, Device Plugins, MIG (Multi-Instance GPU), fractional GPU sharing.  
* **Day 46-50:** Advanced K8s Scheduling for AI: Volcano Scheduler, Kueue (K8s Multi-Cluster/Multi-Queue management), Gang Scheduling (ensuring all workers start together or none start).  
* **Day 51-56 (Weekend inclusive):** Set up a local Minikube/Kind cluster with simulated fake-GPUs or pass-through if available, and deploy a job using a Gang Scheduler concept.

---

## Phase 2: Core Ray & KubeRay Deep Dive (Month 3 & 4\)

### Month 3: Ray Architecture & Core Internals

#### Week 9: Introduction to Ray & Architecture Basics

* **Day 57:** Why Ray? Solving the gap between Python and distributed infrastructure.  
* **Day 58:** Ray Architecture Overview: Head Node vs. Worker Nodes.  
* **Day 59:** Components: GCS (Global Control Store), Raylet, Local Scheduler, Object Store (Plasma).  
* **Day 60:** Ray Tasks (Stateless distributed functions) vs. Ray Actors (Stateful distributed micro-services).  
* **Day 61:** Ray Objects: Pass-by-value vs. Pass-by-reference, Object IDs, and immutable shared memory via Plasma.  
* **Day 62-63 (Weekend):** Set up a local single-node Ray cluster. Write Python code using `@ray.remote` for Tasks and Actors. Inspect the Ray Dashboard.

#### Week 10: Ray Memory Management & Object Store Internals

* **Day 64:** Plasma Object Store Deep Dive: Zero-copy serialization using Apache Arrow.  
* **Day 65:** In-Memory Object eviction, Reference Counting, and Spilling objects to local disk/S3 when Plasma is full.  
* **Day 66:** Debugging Ray OOM errors: Object store OOM vs. Heap memory OOM.  
* **Day 67:** Distributed Object Garbage Collection mechanism in Ray.  
* **Day 68:** Ray Core Configuration tuning parameters for memory-intensive jobs.  
* **Day 69-70 (Weekend):** Write a script that deliberately fills up the Plasma object store and observe how Ray spills objects to disk. Document the logs and behavior.

#### Week 11: Ray Scheduling & Cluster Autoscaler

* **Day 71:** Resource Allocation: Custom resources in Ray (e.g., specifying `num_gpus=1`, `resources={'Custom_Accelerator': 1}`).  
* **Day 72:** Ray Placement Groups: Strict Spread, Strict Pack, Pack, and Spread strategies for distributed training data alignment.  
* **Day 73:** Ray Autoscaler Internals: How the autoscaler monitors the resource demand queue and provisions new nodes.  
* **Day 74:** Ray Cloud Providers integration (AWS, GCP, Azure setup under the hood).  
* **Day 75:** Node dead-marking, heartbeats, and cluster scale-down mechanics.  
* **Day 76-77 (Weekend):** Review the Ray Autoscaler source code layout on GitHub (`ray/python/ray/autoscaler/`). Map out how it triggers a node creation request.

#### Week 12: Ray Fault Tolerance & Resilience

* **Day 78:** Task Lineage Reconstructed: How Ray re-executes lost tasks using object lineage tracking.  
* **Day 79:** Actor Fault Tolerance: Actor restarts, max\_restarts parameters, and state restoration.  
* **Day 80:** GCS (Global Control Store) Fault Tolerance: High availability using external storage or internal sharding.  
* **Day 81:** Network Partitioning handling (Split-brain mitigation in Ray clusters).  
* **Day 82:** Ray Health checks and heartbeat configurations (`ray.init(heartbeat_timeout_ms=...)`).  
* **Day 83-84 (Weekend):** Build a script with a long-running Actor, manually kill the worker process, and configure Ray to automatically recover it without losing previous results.

### Month 4: KubeRay & The Higher-Level Ray Ecosystem

#### Week 13: KubeRay Fundamentals & CRDs

* **Day 85:** Why KubeRay? Bringing Ray's dynamic scaling into Kubernetes control planes.  
* **Day 86:** KubeRay Operator Architecture and Custom Resource Definitions (CRDs): `RayCluster`, `RayJob`, `RayService`.  
* **Day 87:** Deep dive into `RayCluster` Spec: Configuring HeadGroupSpec and WorkerGroupSpecs.  
* **Day 88:** KubeRay Pod Injection: Init containers, sidecars, and mounting shared storage.  
* **Day 89:** KubeRay Autoscaling via K8s HPA or integration with Karpenter/Cluster Autoscaler.  
* **Day 90-91 (Weekend):** Install KubeRay Operator on a local Kubernetes cluster (Kind/Minikube) using Helm. Deploy a basic `RayCluster` YAML.

#### Week 14: Ray Train & Distributed Training Infrastructure

* **Day 92:** Ray Train Architecture: How it wraps PyTorch DDP, DeepSpeed, and Horovod.  
* **Day 93:** `ScalingConfig` parameter configuration: Workers, GPUs per worker, use\_gpu flags.  
* **Day 94:** Ray Train Workers initialization and integration with NCCL inside Kubernetes pods.  
* **Day 95:** Checkpoint Management in Ray Train: Uploading local worker checkpoints directly to cloud storage synchronously/asynchronously.  
* **Day 96:** Handling worker failures inside Ray Train during mid-epoch execution.  
* **Day 97-98 (Weekend):** Run a sample PyTorch training script wrapped inside Ray Train on your local KubeRay cluster. Monitor node pod metrics.

#### Week 15: Ray Data & Streaming Pipelines

* **Day 99:** Ray Data Fundamentals: Datasets, Blocks, and execution streams.  
* **Day 100:** Block-based parallel data processing: Map batches, windowing, and streaming data execution.  
* **Day 101:** Ray Data integration with training: Feeding data from Ray Data into Ray Train without staging to disk.  
* **Day 102:** Optimizing read/write operations from S3/GCS using Ray Data.  
* **Day 103:** Handling out-of-core streaming when text/image datasets exceed cluster memory.  
* **Day 104-105 (Weekend):** Build an end-to-end Ray pipeline: Read a large mock dataset, transform it via Ray Data, and feed it into a Ray Train dummy model.

#### Week 16: Ray Serve & Production Inference Infrastructure

* **Day 106:** Ray Serve Architecture: Controllers, Routers, and Replicas.  
* **Day 107:** Dynamic Request Batching in Ray Serve to optimize GPU utilization.  
* **Day 108:** Multi-model composition: Building pipelines where Model A passes outputs to Model B inside Ray.  
* **Day 109:** Fractional GPU allocation for serving (e.g., running 4 small inference replicas on 1 physical GPU).  
* **Day 110:** KubeRay `RayService` CRD: Zero-downtime upgrades, rollouts, and high-availability serving.  
* **Day 111-112 (Weekend):** Deploy a multi-replica transformer model using `RayService` YAML on your KubeRay setup. Run load testing via Apache Bench or Locust.

---

## Phase 3: Hands-on Production Projects & Tuning (Month 5\)

### Month 5: Scaled Simulation, Fault Injection & Benchmarking

#### Week 17 & 18: Multi-Node AWS/GCP Cluster Deployment (Simulating Production)

* **Days 113-118:** Set up a secure multi-node Kubernetes cluster on AWS (EKS) or GCP (GKE) via Terraform. Use spot instances mixed with on-demand instances.  
* **Days 119-122:** Install KubeRay with multi-node awareness. Configure node taints, tolerations, and affinity rules for proper scheduling on GPU instances vs CPU nodes.  
* **Days 123-126 (Weekend inclusive):** Run a benchmark training run (e.g., fine-tuning a small LLaMA/BERT model) using Ray Train across multiple nodes. Capture network metrics and storage IOPS.

#### Week 19: Fault Injection & Chaos Engineering on Ray Clusters

* **Day 127:** Setting up Chaos tools or scripts to randomly terminate KubeRay worker pods during active training.  
* **Day 128:** Analyzing GCS behavior when head nodes suffer severe latency or network drops.  
* **Day 129:** Benchmarking recovery times: Time taken from worker pod death to resumption of training from the last saved checkpoint.  
* **Day 130:** Mitigating cascading failures: Tuning timeouts and retry thresholds in Raylet configurations.  
* **Day 131:** Profiling slow network interfaces using NCCL environment tuning.  
* **Day 132-133 (Weekend):** Execute a complete "Chaos Training Run". Generate a post-mortem report tracking training logs, state restoration metrics, and loss curves during failure/recovery cycles.

#### Week 20: Cost Optimization & Fine-Grained Performance Profiling

* **Day 134:** Cost metrics compilation: Spot instance termination handling using Ray Autoscaler preemption handling features.  
* **Day 135:** Ray Timeline Profiling: Exporting json traces from Ray and viewing them via `chrome://tracing` or Perfetto.  
* **Day 136:** Identifying execution gaps: Idle GPU time caused by slow data loading or unoptimized Ray Object fetching.  
* **Day 137:** Memory profile analysis: Tracking Ray Heap allocation vs Plasma Object store footprint.  
* **Day 138:** Storage throughput tuning: Benchmarking EFS vs EBS vs S3 streaming performance within Ray Data.  
* **Day 139-140 (Weekend):** Take an unoptimized Ray script, profile it using Ray Timeline, eliminate the data loading bottleneck you identify, and prove a \>30% execution time speedup.

---

## Phase 4: Open-Source Contributions & Interview Prep (Month 6\)

### Month 6: Contributing to Ray/KubeRay & Cracking Niche AI Interviews

#### Week 21 & 22: Open Source Contributions (Targeting Ray Ecosystem)

* **Days 141-145:** Set up the full local build environment for the Ray repository (requiring Bazel, C++, Python setups).  
* **Days 146-150:** Go through Ray and KubeRay GitHub issue trackers. Look for tags like: `good first issue`, `help wanted`, `area/kuberay`, `bug`.  
* **Days 151-154:** Deep dive into KubeRay Go operator controller logic. Look for opportunities to fix bugs, improve error messages, or enhance tests.  
* **Days 155-160:** Submit your first Pull Request (PR) to Ray/KubeRay. It can be documentation fixes, unit tests, or small feature updates. Engage with maintainers for feedback.

#### Week 23 & 24: AI System Design & Interview Preparation

* **Day 161:** System Design Focus: Designing an infrastructure stack to train a 1 Trillion Parameter LLM model across 10,000 GPUs.  
* **Day 162:** Designing a real-time high-throughput serving infrastructure for LLMs handling millions of concurrent users (vLLM, Ray Serve, TensorRT-LLM integration).  
* **Day 163:** Calculating cluster sizing: Estimating necessary memory, networking bandwidth, and compute requirements given specific model specifications and time constraints.  
* **Day 164:** Designing multi-tenant AI Infrastructure: Managing compute allocation, quotas, and priority queue systems across different research teams.  
* **Day 165:** Security considerations in AI Infra: Securing model weights, preventing data leakage during inference pipelines, and cluster access controls (RBAC, Okta configurations).  
* **Day 166-167 (Weekend):** Write out standard mock design answers for: "Design a fault-tolerant distributed training framework similar to Ray."

#### Week 25 & 26: Target Pitching, Resume Overhaul & Mock Interviews

* **Day 168-170:** Refactor your Resume/Portfolio. Highlight Okta/Cohesity production infrastructure experience, combined with Month 3-5 Ray projects, and Month 6 Open-Source PRs. Use terms like *Distributed Compute, NCCL, InfiniBand, KubeRay, GPU Memory Optimization, Fault Tolerance*.  
* **Day 171-173:** Identify open roles at niche AI companies: xAI, AnyScale, OpenAI, Anthropic, Together AI, CoreWeave, Lambda Labs.  
* **Day 174-177:** Reverse engineer profiles of AI Infra Engineers at xAI/AnyScale on LinkedIn to match their technical output.  
* **Day 178-182:** Extensive Mock Interviews focusing on: Distributed Systems theory, Linux OS internals, Advanced Kubernetes networking, Ray cluster scaling, and Python/Golang performance.  
* **Day 183-184 (End of Month 6):** Start applying via warm referrals and open-source networks built through GitHub contributions.

---

## CLOUD-NATIVE MLOPS, OBSERVABILITY & CAPSTONE PROJECTS (Integrated Add-ons)

### Kubernetes GPU Infrastructure, Storage & Control Planes (Cloud-Native Extensions)

* **Topics:** NVIDIA GPU Operator, Device Plugins, MIG, MinIO/Ceph, GPUDirect Storage, KServe / Seldon Core, Canary deployments.  
* **Assignments:** Configure NVIDIA GPU Operator, allocate MIG slices, and setup automated canary deployments for model weights.

### Observability, SRE & Performance Profiling

* **Topics:** DCGM (Data Center GPU Manager) metrics export, OpenTelemetry (OTel) distributed tracing across API Gateway \-\> Router \-\> vLLM worker, defining AI SLIs/SLOs (TTFT, ITL).  
* **Assignment:** Construct a Prometheus \+ Grafana dashboard using DCGM Exporter and vLLM OTel metrics to detect memory fragmentation and GPU throttling.

### Capstone Projects (Integrated Deployment Targets)

* **Capstone 1 (Enterprise-Grade LLM Gateway & Distributed Serving Platform):** Build a production-grade LLM gateway featuring continuous batching, dynamic model loading/unloading, KV-cache eviction policies, Prometheus observability, and Ray autoscaling on Kubernetes.  
* **Capstone 2 (High-Throughput Custom Multi-Modal Inference Engine):** Build a multi-stage pipeline (Vision Model \+ LLM) deployed on Triton Inference Server using dynamic batching, shared memory IPC, and custom C++/CUDA preprocessing kernels.

---

