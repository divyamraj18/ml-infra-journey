# 📅 Day 37: Automated Health Checks, Node Fencing, & Silent Data Corruption (SDC)

Today’s focus is on building resilient production infrastructure that proactively detects bad hardware before and during LLM training. We will explore GPU health diagnostic suites, hardware isolation via **Node Fencing**, and techniques for identifying and mitigating **Silent Data Corruption (SDC)**.

## 1. Automated Pre-Flight & In-Flight Health Checks

In large multi-node GPU clusters, starting or continuing a training run without testing hardware integrity leads to wasted compute cycles, frequent job crashes, or corrupted weights.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Lifecycle of Cluster Diagnostics                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Node Provisioned ──► 2. Pre-Flight Diagnostic (DCGM/NCCL Tests)         │
│                                 │                                           │
│                     ┌───────────┴───────────┐                               │
│                     ▼                       ▼                               │
│                [ PASSED ]              [ FAILED ]                           │
│                     │                       │                               │
│                     ▼                       ▼                               │
│         3. Added to Active Pool    4. Node Fenced (Isolated for Repair)     │
│                     │                                                       │
│                     ▼                                                       │
│         5. In-Flight Health Monitor (Runtime Metrics / NaN Checks)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Health Testing Mechanisms:

1. **NVIDIA DCGM Diagnostics (Data Center GPU Manager):**
   * Runs standardized tests ranging from quick software health verification (`Level 1`) to intense hardware stress tests (`Level 3`), evaluating PCIe bandwidth, memory integrity, and NVLink interconnect stability.

2. **NCCL Bandwidth Tests (`nccl-tests`):**
   * Benchmarks all-reduce and all-gather operations across intra-node NVLink and inter-node InfiniBand networks. Nodes failing to meet peak throughput thresholds (e.g., `< 350 GB/s` inter-node) are flagged immediately.

3. **In-Flight Runtime Monitoring:**
   * Continuously tracks per-GPU metrics such as PCIe error counters, memory ECC errors, GPU throttles (due to thermal or power limits), and NVLink error states without stopping training.

---

## 2. Node Fencing Mechanics

When a node experiences hardware degradation or fails pre-flight checks, it must be removed from the active scheduling pool instantly to prevent downstream training failures.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Node Fencing Flowchart                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Cluster Node Pool                                                          │
│  [Node A: Healthy]   [Node B: High ECC Errors]   [Node C: Healthy]          │
│                             │                                               │
│                             ▼                                               │
│             Node Fencing Trigger (Automated Agent)                          │
│                             │                                               │
│                             ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Set Node State to `DRAINING` / `UNSCHEDULABLE` (K8s Cordon / Slurm) │  │
│  │ 2. Evict Active Workloads / Terminate Sync Processes                  │  │
│  │ 3. Send Alert to Hardware Ops & Trigger Automated Repair Diagnostics  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **Cordoning:** Marks the node so the scheduler (Slurm or Kubernetes) assigns no new pods/jobs to it.
* **Drain & Isolate:** Graciously evicts or terminates running tasks, isolating the host on the network level if necessary to prevent bad packets or deadlocked TCP connections from affecting remaining workers.

---

## 3. Detecting & Mitigating Silent Data Corruption (SDC)

Unlike explicit hardware faults that emit system panics, **Silent Data Corruption (SDC)** occurs when micro-faults in Tensor Cores or HBM flip bits silently, introducing incorrect mathematical results into model calculations.

### A. Manifestations of SDC in LLM Training

* **Gradient Explosions / NaNs:** A corrupted floating-point multiply-accumulate operation yields `NaN` or `Inf`, spreading rapidly through collective communications to ruin all model weights.
* **Loss Spikes & Degradation:** Small bit flips alter layer activations, leading to unexplainable divergence in loss curves without crashing the process.

### B. Detection & Prevention Protocols

1. **Gradient & Activation Assertions:**
   ```python
   # In-band check before optimizer step
   def validate_gradients(model):
       for name, param in model.named_parameters():
           if param.grad is not None:
               if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                   raise ValueError(f"SDC Detected: NaN/Inf in gradient of {name}")
   ```

2. **Deterministic Golden Run Verification:**
   * Periodically running a synthetic fixed-seed micro-batch across GPUs and verifying checksums of activation outputs against known "golden" reference values.

3. **Memory ECC (Error-Correcting Code):**
   * Enabling hardware ECC on HBM to detect and automatically correct single-bit errors while reporting uncorrectable double-bit errors to DCGM.

---

## 4. Summary Checkpoints for Day 37

1. **Pre-Flight Health Checks:** Running DCGM and NCCL throughput benchmarks before job launch prevents bad hardware from crashing large-scale training jobs.
2. **Automated Fencing:** Instantly isolating failing or degraded nodes (cordoning/draining) maintains cluster health and uptime.
3. **SDC Mitigation:** In-band NaN checks and periodic deterministic output verification protect model weights against silent bit-flips and corrupted hardware calculations.