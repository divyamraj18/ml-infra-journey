# 📅 Day 38: Slurm Workload Manager vs. Kubernetes for AI Workloads

Today’s focus is introducing the two primary cluster management ecosystems used for AI and HPC workloads: **Slurm Workload Manager** and **Kubernetes (K8s)**. We will examine how jobs are scheduled, resources are allocated, and state is managed across these platforms.

## 1. Overview of Cluster Orchestration in AI

Large-scale distributed training and inference require a control plane to allocate GPUs, manage network topology, handle node failures, and queue incoming jobs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Cluster Orchestration Control Layer                   │
├─────────────────────────────────────────────────────────────────────────────┤
│   Job Submission (User Scripts / MPI / Helm / Custom Schedulers)             │
│                                     │                                       │
│                                     ▼                                       │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │ Scheduler Engine (Slurm Controller / Kubernetes API + Scheduler)      │ │
│   └───────────────────────────────────┬───────────────────────────────────┘ │
│                                       │                                     │
│            ┌──────────────────────────┴──────────────────────────┐          │
│            ▼                                                     ▼          │
│   [ Compute Node 01 ]                                   [ Compute Node 02 ] │
│   ┌────────────────────────┐                            ┌─────────────────┐ │
│   │ GPU 0 | GPU 1 | GPU 2  │                            │ GPU 0 | GPU 1   │ │
│   └────────────────────────┘                            └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Slurm Workload Manager: The HPC Benchmark

**Slurm (Simple Linux Utility for Resource Management)** originated in High-Performance Computing (HPC) environments. It operates directly on bare-metal hardware.

### Key Architectural Concepts:
* **`slurmctld` (Central Controller):** Manages queues, monitors node states, and allocates hardware resources to queued jobs.
* **`slurmd` (Node Daemon):** Runs on every compute worker node, executing tasks and reporting hardware metrics.
* **Batch Directives (`sbatch`, `srun`, `salloc`):** Users submit jobs via shell scripts specifying raw node counts, tasks-per-node, and GPU requirements.

```bash
#!/bin/bash
#SBATCH --job-name=llm_train_node128
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=gpu-h100

srun python -m torch.distributed.run train.py
```

### Advantages of Slurm:
1. **Low Overhead:** Direct access to host hardware and kernel interfaces without container network translation barriers.
2. **Native MPI & InfiniBand Integration:** Excellent support for direct inter-node communications over high-speed interconnects.
3. **Topology Awareness:** Schedules jobs based on switch infrastructure layout to minimize network hops between communicating ranks.

## 3. Kubernetes (K8s): The Cloud-Native Engine

**Kubernetes** was built for web-scale container orchestration, prioritizing declarative API management, dynamic service scaling, and microservices.

### Key Architectural Concepts:
* **Control Plane (API Server, `etcd`, `kube-scheduler`):** Maintains cluster state declaratively via JSON/YAML manifests.
* **`kubelet` & Container Runtime (Containerd/CRI-O):** Executes workloads wrapped inside containerized Pods.
* **Custom Resource Definitions (CRDs):** Operators like `Kubeflow Training Operator` (`PyTorchJob`) adapt K8s to handle multi-pod distributed training setups.

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llm-train-job
spec:
  pytorchReplicaSpecs:
    Worker:
      replicas: 16
      template:
        spec:
          containers:
          - name: pytorch
            image: llm-train:v1.0
            resources:
              limits:
                nvidia.com/gpu: 8
```

### Advantages of Kubernetes:
1. **Immutable Containerization:** Standardized environments across development and production eliminate system-dependency mismatches.
2. **Rich Ecosystem:** Seamless integration with cloud logging, service meshes, dynamic ingress controllers, and auto-scalers.
3. **Multi-Tenancy & Isolation:** Strong namespace and RBAC boundaries tailored for diverse microservice deployments.

## 4. Architectural Comparison: Slurm vs. Kubernetes

| Feature | Slurm Workload Manager | Kubernetes (K8s) |
| :--- | :--- | :--- |
| **Primary Domain** | HPC / Supercomputing | Cloud-Native / Microservices |
| **Execution Layer** | Bare-metal (Processes / Modules) | Containers (Docker / OCI / Pods) |
| **Network Overhead** | Near-zero (Direct RDMA / InfiniBand) | Require CNI (Calico/Cilium) or HostNetworking |
| **Gang Scheduling** | Native out-of-the-box | Requires plugins (Volcano / Kueue) |
| **Storage Integration** | Direct POSIX (Lustre / WekaIO) | CSI Drivers (PersistentVolumes) |
| **Primary Use-Case** | Massive Batch Training Runs | Model Inference, Microservices, Hybrid Workflows |

## 5. Summary Checkpoints for Day 38

1. **Slurm Focus:** Built for maximum bare-metal throughput, hardware topology matching, and HPC batch jobs.
2. **Kubernetes Focus:** Built for containerized isolation, declarative state management, and ecosystem integration.
3. **Emerging Trend:** Enterprise AI infrastructure frequently bridges both, running batch training on Slurm/custom K8s setups while serving inference workloads entirely on Kubernetes.