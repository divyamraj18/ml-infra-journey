# Day 10: RDMA

Day 9 said a gradient should not pass through the CPU TCP stack. **RDMA (Remote Direct Memory Access)** is that mechanism. The NIC writes into memory on the remote machine. The remote CPU does not run `recv()`.

## Two machines, a NIC in the middle

Each side has an **RNIC**. On a DGX that is a ConnectX. The RNIC knows the physical address of memory that userspace registered, so it can DMA after the kernel has granted that mapping.

Three objects to keep:

| Object | What it is |
|---|---|
| **Queue Pair (QP)** | A Send Queue plus a Receive Queue. The endpoint of one connection. The RDMA cousin of a TCP socket, except the payload does not enter the kernel. |
| **Completion Queue (CQ)** | The NIC writes here when a work request finishes. The CPU polls it. An interrupt is not required. |
| **Memory Region (MR)** | Pages pinned for the NIC. **lkey** is for local access. **rkey** is for a remote writer. Without the rkey, the remote NIC cannot touch your memory. |

A **Protection Domain** groups those keys so another process's QP cannot use your MR.

The CPU still does setup: pin pages, create the QP, send the remote side your rkey and address. Those control messages are small. **Bootstrap** is often still TCP. The data path after that is RDMA. Do not confuse bootstrap with the data path.

## Verbs

| Verb | Who is active | In an LLM job |
|---|---|---|
| **SEND / RECV** | Both sides. The receiver posts a buffer first. The sender SENDs into it. | Small control messages. |
| **WRITE** | Sender only. It writes to a remote address plus rkey. The remote CPU is not involved. | Large gradient chunks. NCCL's common data path. |
| **READ** | Sender only. It pulls from the remote side. | Less common. Used when the sender wants to pull. |
| **Atomic** | Fetch-and-add or compare-and-swap, executed on the NIC. | Locks and counters. Not the bulk of a collective. |

WRITE is **one-sided**. The remote application does not get a callback for that packet. The NIC DMA'd it, and the memory changed. The sender learns completion from its CQ.

## Kernel bypass

On TCP, every `send()` is a syscall. The kernel copies the buffer and runs the TCP state machine.

On RDMA, the process rings the QP **doorbell** with an MMIO write. The work request lands on the NIC's queue. A small message does not take a syscall. The CPU does not read the payload. Latency is the NIC, PCIe, and switch hops, not kernel scheduling.

This does not mean the CPU is idle. The CPU builds the work request and polls the CQ. What the CPU does not do: copy the payload, checksum it, or run `recv()` on the remote side.

## GPUDirect RDMA

So far, RDMA was about host DRAM. The gradient is in HBM. Without **GPUDirect RDMA**, the NIC can DMA only host DRAM, so you first `cudaMemcpy` from HBM to DRAM, then RDMA. That is PCIe twice.

With GPUDirect RDMA, GPU memory is also an MR. The NIC's DMA engine reaches HBM through a peer mapping between the GPU and the NIC. On older stacks the kernel module is `nvidia-peermem`. On newer drivers that support lives inside the driver.

```text
TCP
  HBM --PCIe-- DRAM --kernel-- NIC --wire-- NIC --kernel-- DRAM --PCIe-- HBM

GPUDirect RDMA
  HBM --DMA-- NIC --wire-- NIC --DMA-- HBM
```

The NIC and the GPU have to be **close**. The same PCIe switch (**PIX**) is the best case. Across CPU sockets (**SYS**), GPUDirect may be disabled, because the DMA would cross QPI or UPI. `nvidia-smi topo -m` prints that distance. The legend is on Day 12.

## Latency and bandwidth are different

Same split as Day 2.

A **small message**, such as a sync flag. Time = PCIe + NIC + one switch hop. Bandwidth does not matter. 64 bytes on a 400 Gb/s link is nanoseconds of wire time, but setup can take microseconds. That message is **latency-bound**.

A **large message**, 14 GB of gradients. Time ≈ size / bandwidth. On a 400 Gb/s rail that is a fraction of a second. The CPU is already off the path. The limit is the optical port. That message is **bandwidth-bound**.

NCCL uses different protocols for the two. **LL** (low latency) for small messages. **Simple** for large ones. **LL128** sits in between. Those names show up in the Day 12 log. The wrong protocol makes a small AllReduce pay the large-message overhead.

## Rule for today

RDMA means the NIC WRITE or READ on a remote MR, using an rkey, without the remote CPU calling `recv()`. GPUDirect extends that DMA into HBM. The TCP CPU copy is not on this path. A small message is limited by latency. A large gradient is limited by bandwidth.
