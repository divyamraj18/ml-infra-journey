# 📅 Day 33-34: System Architecture — Storage Design for a 100B Parameter Model Cluster

This module presents an end-to-end infrastructure architecture designed to support continuous training of a **100 Billion parameter LLM** on a 1,024 GPU cluster (128x 8-GPU nodes), optimizing for high-throughput data streaming and low-overhead burst checkpointing.

---

## 1. Workload Specification & Memory Footprint

### A. Model Parameters & Optimizer States
For a 100B model trained in FP16/BF16 using standard AdamW:
* **Model Parameters (FP16/BF16):** $\text{100B} \times 2 \text{ bytes} = 200 \text{ GB}$
* **Gradients (FP16/BF16):** $\text{100B} \times 2 \text{ bytes} = 200 \text{ GB}$
* **AdamW Optimizer States (FP32):**
  * Master Weights: $\text{100B} \times 4 \text{ bytes} = 400 \text{ GB}$
  * First Moment (Momentum): $\text{100B} \times 4 \text{ bytes} = 400 \text{ GB}$
  * Second Moment (Variance): $\text{100B} \times 4 \text{ bytes} = 400 \text{ GB}$
* **Total Static Checkpoint Size per Snapshot:** $\mathbf{1.6 \text{ TB}}$

### B. I/O Workload Profiles
1. **Continuous Input Streaming (Read Heavy):**
   * Target Consumption Rate: ~1.5 to 2.0 TB of tokenized data per day across ranks.
   * Pattern: Sequential streaming of Tar-sharded datasets via WebDataset.
2. **Periodic Checkpointing (Write Heavy):**
   * Target Frequency: Every 2 hours or every 1,000 steps.
   * Pattern: Sudden burst write of **1.6 TB** distributed across compute nodes within a $< 30 \text{ second}$ target window (requires $\approx 55 \text{ GB/s}$ sustained write throughput).

---

## 2. Multi-Tier Storage System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 1,024 GPU Training Cluster                                      │
│   [ Node 01 ]        [ Node 02 ]        [ Node 03 ]       ...        [ Node 128 ]                 │
│   (Local NVMe RAM)   (Local NVMe RAM)   (Local NVMe RAM)             (Local NVMe RAM)             │
└───────┬──────────────────────┬──────────────────┬───────────────────────────┬────────────────────┘
        │                      │                  │                           │
        │ Dual 200Gbps NDR     │ Dual 200Gbps NDR │ Dual 200Gbps NDR          │ Dual 200Gbps NDR
        ▼                      ▼                  ▼                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: High-Performance Parallel Storage (WekaIO / Lustre NVMe Appliance)                     │
│  - Total Capacity: 500 TB High-Speed NVMe Storage Pool                                           │
│  - Aggregate Throughput: 250 GB/s Sequential Read / Write                                        │
│  - Protocol: GPUDirect Storage (GDS) over InfiniBand RoCEv2                                      │
└────────────────────────────────────────────────────────┬─────────────────────────────────────────┘
                                                         │
                                                         │ Background Async Sync Engine
                                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: Remote Capacity Storage (Cloud Object Storage - AWS S3 / MinIO Cluster)                 │
│  - Total Capacity: Multi-Petabyte Cold Storage                                                  │
│  - Role: Long-term checkpoint storage, archival & global dataset repository                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline & Data Loading Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Data Ingestion Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. Dataset Sharding:                                                    │
│    Data pre-processed into WebDataset TAR shards (512MB per shard).     │
│                                                                         │
│ 2. Local Node Caching (Tier 0):                                         │
│    Alluxio / Ray Data workers stream shards directly into node-local    │
│    Shared Memory (/dev/shm) or local NVMe scratch space.                │
│                                                                         │
│ 3. Zero-Copy Reader:                                                    │
│    DataLoader reads contiguous memory blocks without Python pickle      │
│    serialization overhead using shared memory IPC handles.              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Checkpoint Flow Architecture (PyTorch DCP + Asynchronous Write)

```
[ GPU Cluster Ranks ]
       │
       │ 1. Save state_dict in CPU Pinned Memory (~1.5s pause)
       ▼
[ Node Host Memory ] ─── (Rank Training Resumes Immediately) ───► GPU Step Continues
       │
       │ 2. Non-blocking Background Async Thread Worker
       ▼
[ GPUDirect Storage (GDS) / NVMe Fabrics ]
       │
       │ 3. Write chunked sharded files to Tier 1 WekaIO / Lustre System
       ▼
[ Tier 1 NVMe Parallel Storage ]
       │
       │ 4. Background Sync Job (Low priority)
       ▼
[ Tier 2 S3 Object Store Archive ]
```

### Key Architectural Trade-offs Matrix

| Feature | Direct-to-S3 Saving | Traditional POSIX NFS | Proposed Multi-Tier PFS + DCP |
| :--- | :--- | :--- | :--- |
| **Checkpoint Duration (Block)** | 5 - 15 minutes | 3 - 8 minutes | **< 3 seconds (Blocking)** |
| **I/O Bandwidth Bottleneck** | S3 API rate limit | NFS Head Controller Lock | None (Parallel NVMe Striping) |
| **Data Recovery Time (MTTR)** | High (Download overhead) | Medium | Fast (Local Tier 1 Read) |
| **Storage Cost Optimization** | Low | High | Optimal (Tier 1 Flash + Tier 2 Object) |

---

## 5. Summary Checkpoints for Day 33-34

1. **Storage Tiering:** Separating compute-adjacent Tier 1 flash (WekaIO/Lustre) from capacity Tier 2 storage (S3) provides optimal price-performance balance.
2. **Asynchronous Checkpointing:** Staging state dumps into host pinned RAM isolates GPU execution, cutting checkpoint blocking duration from minutes to under 3 seconds.
3. **WebDataset Ingestion:** Converting raw data into sharded tar streams avoids POSIX file locking and saturates system memory buses.