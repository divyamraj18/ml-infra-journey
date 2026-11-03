# 📅 Day 46-50: Advanced K8s Scheduling for AI (Volcano & Kueue)

Aapka yeh module **Advanced Kubernetes Scheduling** par focus karta hai. Default Kubernetes scheduler (`kube-scheduler`) individual pods ko ek-ek karke independent entities ki tarah schedule karta hai. Lekin distributed LLM training jobs (jaise PyTorch DDP ya Megatron-LM) ke liye yeh approach fail ho jati hai, kyunki agar ek bhi pod schedule nahi ho paya, toh baaki pods hang ho jaate hain (Deadlock / Dead Worker state). Is problem ko solve karne ke liye hum **Gang Scheduling** aur advanced batch schedulers jaise **Volcano** aur **Kueue** ka use karte hain.

## 1. The Default Kubernetes Scheduler Limitation: Gang Scheduling Deficit

Distributed AI training jobs require all worker ranks to start simultaneously to establish TCP/NCCL communication rings.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   The Distributed Scheduling Deadlock                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Job Requires: 4 Pods (Worker-0 to Worker-3)                                │
│                                                                             │
│  Default Scheduler Action:                                                  │
│  - Allocates Worker-0, Worker-1, Worker-2                                   │
│  - Node out of GPU memory for Worker-3 ──► Pod 3 remains Pending.           │
│  - Worker-0 to 2 wait indefinitely for Worker-3 (NCCL Timeout Deadlock).    │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Gang Scheduling (All-Or-Nothing Scheduling):

* **Concept:** A job is treated as a single atomic unit ("Gang"). Either *all* pods in the gang are scheduled concurrently onto available cluster resources, or *none* of them are scheduled.

* **MinAvailable Rule:** If a job requires 16 GPUs (2 nodes of 8 GPUs), the scheduler ensures no resources are locked down until all 16 slots are simultaneously free.

## 2. Volcano Scheduler Architecture

**Volcano** is a native cloud-batch system designed under CNCF for HPC, deep learning, and big data workloads on Kubernetes.

```
┌─────────────────────────────────────────────────────────────────────────────┤
│                       Volcano Architecture Overview                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Custom Resource: `PodGroup` (Defines minMember = 16, minTaskMember)        │
│                                       │                                     │
│                                       ▼                                     │
│  Volcano Central Controller / Scheduler Plugins                             │
│  ├── 1. Gang Plugin (Ensures atomic all-or-nothing placement)              │
│  ├── 2. Topology Plugin (Collocates pods on same switch/InfiniBand fabric)  │
│  └── 3. Overcommit / Queue Plugin (Manages multi-tenant fairness)          │
│                                       │                                     │
│                                       ▼                                     │
│  Execution Nodes (Allocates pods simultaneously across worker nodes)        │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Key Volcano Plugins:

1. **Gang Plugin:** Enforces atomic scheduling batches using Kubernetes `PodGroup` CRDs.

2. **Topology Plugin:** Ensures communication-heavy pods (like Tensor Parallelism ranks) are scheduled on the same NVLink domain or InfiniBand leaf switch to minimize network latency hops.

3. **Queue Plugin:** Enables hierarchical multi-tenancy where teams (e.g., Research, Production, Fine-Tuning) get guaranteed shares of GPU capacity with borrow/preemption mechanisms.

## 3. Kubernetes Kueue: Native Job Queueing & Resource Management

While Volcano replaces or extends low-level scheduling loops, **Kueue** is a modern upstream Kubernetes project focusing on **job queueing, quota management, and workload admission** without modifying the core scheduler.

### How Kueue Works:

* **ClusterQueue:** Defines compute resources (e.g., H100 GPU pools) available to groups of namespaces, handling borrow/lend policies between teams.

* **LocalQueue:** Maps user namespaces to specific ClusterQueues.

* **Workload Admission:** When a PyTorchJob or Kubeflow training job is submitted, Kueue intercepts the creation request, holds it in a pending queue until cluster capacity is guaranteed, and then admits it for scheduling by the standard scheduler.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Kueue Admission Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  User Submits PyTorchJob ──► Kueue Admission Controller (Holds in Queue)    │
│                                       │                                     │
│                                       ▼                                     │
│              Cluster Capacity Check (Sufficient Free GPUs?)                 │
│                      ┌────────────────┴────────────────┐                    │
│                      ▼                                 ▼                    │
│                 [ YES ]                           [ NO ]                    │
│                      │                                 │                    │
│                      ▼                                 ▼                    │
│         Kueue Admits Workload                 Remains Queued (FIFO /     │
│         (Passes to Scheduler)                 Priority Order)               │
└─────────────────────────────────────────────────────────────────────────────┘

```

## 4. Summary Checkpoints for Days 46-50

1. **Gang Scheduling:** Essential for distributed training to prevent distributed deadlocks by ensuring atomic, all-or-nothing pod placement.

2. **Volcano Scheduler:** Provides robust CNCF batch scheduling features, including topology awareness, gang execution, and multi-tenant queuing.

3. **Kueue:** A cloud-native queue manager that handles fair-share resource quotas and workload admission before tasks hit the Kubernetes scheduler.