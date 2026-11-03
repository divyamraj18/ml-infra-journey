# 📅 Day 40-41: Case Studies — Cluster Infrastructure at Meta & OpenAI Scale

This module provides a deep-dive analysis of real-world infrastructure architectures built by industry leaders to train frontier models (such as Llama 3 and GPT-4). We will examine how **Meta** uses heavily customized Slurm architectures for bare-metal performance, and how **OpenAI** (in partnership with Microsoft Azure) scaled Kubernetes to manage massive distributed training runs.

## 1. Meta's Llama 3 Infrastructure (Custom Slurm + Grand Teton Architecture)

Meta built two 24,576 GPU clusters (powered by NVIDIA H100s) to train the **Llama 3** model family. Their design prioritizes **bare-metal execution speed**, zero-overhead communications, and maximum storage throughput.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Meta's Llama 3 Cluster Footprint                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Compute Engine : 24,576 NVIDIA H100 GPUs (80GB HBM3)                       │
│  Orchestration : Slurm + Custom Meta Management Plugins                     │
│  Network Fabric: RoCE v2 (Arista 400G Switches) + NVIDIA Quantum-2 IB       │
│  Storage Layer : Tectonic Distributed FS + WekaIO NVMe Burst Buffers        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Choices:

1. **Bare-Metal Slurm Orchestration:**
   * Meta opted for Slurm to avoid container translation penalties, API-server limits, and overlay network latencies across 24,000+ ranks.
   * Custom extensions were developed for Slurm to manage multi-tenant queueing, automated node drain/fencing triggers, and dynamic job rescheduling upon hardware failures.

2. **Dual Interconnect Strategy:**
   * **Cluster A:** Built on **RoCE v2 (RDMA over Converged Ethernet)** using Arista switches. Required extensive tuning of Priority Flow Control (PFC) and Explicit Congestion Notification (ECN) to eliminate packet drops.
   * **Cluster B:** Built on **NVIDIA Quantum-2 InfiniBand**. Provided out-of-the-box adaptive routing and telemetry.

3. **Storage Pipeline (Tectonic & WekaFS):**
   * Meta utilized **Tectonic** (their internal exabyte-scale distributed object store) backed by local **WekaIO** Flash arrays acting as high-speed burst buffers to handle concurrent checkpointing without stalling GPU computation.

---

## 2. OpenAI’s Infrastructure (Large-Scale Kubernetes on Azure)

OpenAI took a cloud-native approach, scaling **Kubernetes** on Microsoft Azure to orchestrate tens of thousands of GPUs across thousands of virtual nodes for model training and serving.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OpenAI Kubernetes Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Compute Layer : Azure Compute Instances (NDv4 Series - A100/H100)           │
│  Orchestration : Kubernetes (Significantly Patch-Tuned Control Plane)       │
│  Scheduler     : Custom Gang Scheduler / Kueue-like Queue Management        │
│  Network Fabric: Azure InfiniBand (SR-IOV Direct Pass-Through)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Overcoming Kubernetes Bottlenecks at 10,000+ Nodes:

1. **Control Plane Scaling & `etcd` Optimization:**
   * Standard K8s control planes degrade when thousands of nodes report pod statuses simultaneously. OpenAI split `etcd` datastores into distinct instances (one for events, one for pod states) and implemented aggressive client-side caching.

2. **HostNetworking & SR-IOV for GPU Traffic:**
   * To prevent Kubernetes Container Network Interfaces (CNIs) from bottlenecking GPU-to-GPU communications, OpenAI bypassed standard pod network bridges. Pods run in `hostNetwork: true` mode, with InfiniBand interfaces attached via **Single Root I/O Virtualization (SR-IOV)** directly to the CUDA runtime.

3. **Custom Gang Scheduling:**
   * OpenAI replaced default K8s scheduling mechanisms with a custom gang-scheduler. It ensures all 10,000+ pod replicas required for a distributed training job are allocated simultaneously; if a single pod fails to place, the entire batch waits in queue, preventing deadlock.

---

## 3. Comparative Architecture Summary

| Dimension | Meta (Llama 3 Infrastructure) | OpenAI (Azure Cluster Infrastructure) |
| :--- | :--- | :--- |
| **Primary Scheduler** | Slurm (Customized HPC Extensions) | Kubernetes (Cloud-Native Tuned API Server) |
| **Deployment Model** | On-Prem Bare-Metal Data Centers | Cloud Virtualization (Microsoft Azure) |
| **Interconnect** | Dual: RoCE v2 (Ethernet) + InfiniBand | Azure InfiniBand (SR-IOV Passthrough) |
| **Containerization** | Minimal / Host Process Isolation | Strict OCI Containers (Docker/Containerd) |
| **Gang Scheduling** | Native Slurm Allocation Mechanics | Custom Kubernetes Operator / Controller |
| **Failure Recovery** | Slurm Re-queueing + PyTorch DCP | Custom K8s Pod Eviction + Dynamic Resets |

---

## 4. Summary Checkpoints for Day 40-41

1. **Meta Strategy:** Prioritizes bare-metal hardware efficiency, using customized Slurm and deep OS network tuning (RoCE v2/InfiniBand) to saturate 24k+ GPUs.
2. **OpenAI Strategy:** Proves Kubernetes can scale to massive distributed training runs when paired with `etcd` partitioning, `hostNetwork` IB pass-through, and custom gang schedulers.
3. **Core Lesson:** At frontier scale, the choice between Slurm and Kubernetes requires heavy customized engineering layer additions to handle failure management, network passthrough, and sync bottlenecks.