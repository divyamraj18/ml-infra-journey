# Day 8: NVLink, NVSwitch, coalescing rules, bank conflicts, tiled GEMM

Week 1 followed traffic inside one GPU: DRAM, then PCIe, then HBM, then SRAM, then registers. This week asks two new questions. How do 8 GPUs inside one box talk to each other, and how should one GPU read global memory and shared memory so it does not waste bandwidth?

## Intra-node: PCIe, NVLink, NVSwitch

**Intra-node** means one machine, one chassis. A DGX H100 is the example: 8 H100 GPUs in one box.

**PCIe** connects a GPU to the CPU and to the NIC. Gen4 x16 is about 32 GB/s per direction. Gen5 x16 is about 64 GB/s. That is too thin for GPU-to-GPU traffic. Two H100s talking only over PCIe leave their 3.35 TB/s of HBM idle, waiting on that link.

**NVLink** is a direct GPU-to-GPU link.

| GPU | NVLink generation | Bidirectional bandwidth |
|---|---|---|
| A100 | 3rd | ~600 GB/s |
| H100 | 4th | ~900 GB/s |
| B200 | 5th | ~1.8 TB/s |

An H100 has 18 NVLink 4 links. One link is about 25 GB/s per direction. `18 × 25 × 2 = 900 GB/s` bidirectional. That is the Day 6 number.

**NVSwitch** is a crossbar chip. Without a switch, NVLink only reaches a few neighboring GPUs. The switch connects every GPU to every other GPU at full bandwidth.

DGX H100:

- 8 H100 GPUs
- 4 third-generation NVSwitch chips
- A GPU is not wired directly to another GPU. Each GPU plugs its NVLink links into the switches.
- Any pair, including GPU3 and GPU7, can talk through NVSwitch at about 900 GB/s.

This is only inside the box. NVLink stops at the chassis. The next day, InfiniBand, starts outside it.

```text
        NVSwitch  NVSwitch  NVSwitch  NVSwitch
          |  \      |  \      |  \      |  \
        GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
          |    |    |    |    |    |    |    |
         PCIe / NIC   ... these links leave the box. They are not NVLink.
```

## Coalescing rules

Day 4: one warp, 32 threads, one **128-byte sector**. 32 floats × 4 bytes = 128 bytes. Best case is one transaction, and every byte is useful.

Rules for today:

1. **The warp issues together.** Hardware merges 32 separate loads into as few 128-byte sectors as it can. The threads do not merge anything themselves.
2. **Alignment.** A sector starts on a 128-byte boundary. If the warp's first address is not 128-byte aligned, those 128 bytes split across two sectors. Two transactions, and some bytes are wasted.
3. **Stride.** Thread `t` reads element `t * stride`.
   - `stride = 1`: consecutive floats. One sector. Coalesced.
   - `stride = 2`: every other float. Half of the 128 bytes are useful. 50% of the bandwidth is wasted.
   - `stride = 32`: each thread hits a different sector. 32 transactions. The bus moves `32 × 128` bytes, and the kernel only uses `32 × 4` bytes.
4. **Row-major matrix.** Memory stores one row, then the next row. A warp that reads consecutive columns of one row has stride 1. A warp that reads one column has stride equal to the row width. On a 4096-wide row that stride is 4096. That access is uncoalesced.

Assignment 1.1 followed this rule: thread `i` read `a[i]`, stride 1. The 66.84 GiB/s on the T4 was not lost to broken coalescing. That run was far from peak because the buffer was small and each load was a `float`, not a `float4`.

## Shared memory bank conflicts

Shared memory is SRAM, not HBM. A warp still reads it in one cycle only when 32 threads hit **32 different banks**.

There are 32 banks. Each bank is 4 bytes wide, one 32-bit word.

```text
word = byte_offset / 4
bank = word % 32
```

| Access | Result |
|---|---|
| Thread `t` reads `smem[t]` | 32 different banks. One transaction. |
| Every thread reads the same address `smem[0]` | **Broadcast.** Not a conflict. One value is delivered to the whole warp. |
| Thread `t` reads `smem[t * 32]` | Every address lands in bank 0, and the words are different. **32-way conflict.** 32 serial transactions. |
| Some threads share an address, the rest hit different banks | Multicast. Only the distinct banks matter. |

A conflict does not make the answer wrong. It serializes SRAM bandwidth. The kernel stays correct and gets slow.

**Padding** is the usual fix. A tile that is 32 columns wide, read down a column, maps `row * 32 + col` onto the same bank every row, because `32 % 32 = 0`. Store it as if the row were 33 wide:

```text
index = row * (32 + 1) + col
```

The same column in the next row is 33 words away. `33 % 32 = 1`, so the next row hits the next bank. One extra column is wasted, and the conflicts go away.

On H100, shared memory is configurable up to **228 KB per SM**. On A100 it is **164 KB**. The tile has to fit. If it does not, the launch fails or occupancy drops.

## Tiled matrix multiplication

`C[M, N] = A[M, K] × B[K, N]`.

`C[i, j]` is a dot product of length K: row `A[i, :]` with column `B[:, j]`.

**Naive.** One thread computes one `C[i, j]` and reads the `k` loop from global memory. If a warp owns consecutive `j`, then `B[k][j]` is coalesced and `A[i][k]` is a broadcast. Coalescing can be fine. The problem is reuse.

One element of `A` is used by N output columns. At 4096×4096, N is 4096. `A` itself is `4096 × 4096 × 4 = 64 MB`. A100 L2 is 40 MB. H100 L2 is 50 MB. The whole of `A` does not stay in L2, so that element comes back from HBM again and again. Arithmetic intensity looks high on paper. On the machine the kernel is memory-bound.

**Tiling** forces the reuse into SRAM. A block computes one `BM × BN` tile of `C`. K is walked in steps of `BK`.

```text
acc = 0
for k0 in 0, BK, 2*BK, ... K:
    coalesced load A[i0 : i0+BM, k0 : k0+BK]  -> shared As
    coalesced load B[k0 : k0+BK, j0 : j0+BN]  -> shared Bs
    __syncthreads()          # every thread has finished the load
    for kk in 0 .. BK-1:
        acc += As[local_row][kk] * Bs[kk][local_col]
    __syncthreads()          # nobody reads the old As while the next tile loads
C[i, j] = acc
```

`__syncthreads()` waits for every thread in the block. It is not a warp barrier. Two blocks do not share SRAM, so the barrier is only inside one block.

Small picture, `BM = BN = BK = 2`. The block owns the top-left 2×2 of `C`.

```text
k0 = 0:  As = A[0:2, 0:2]    Bs = B[0:2, 0:2]
k0 = 2:  As = A[0:2, 2:4]    Bs = B[2:4, 0:2]
```

`A[0][0]` enters shared memory once and both output columns of that tile use it. It is not fetched from global memory again until the next K step. The naive loop left that reuse to L2.

The global load is still row-wise, so it can stay coalesced. Column-wise reuse happens inside shared memory, which is where bank conflicts show up. `BK = 32` with no padding makes a column read a 32-way conflict. The fix is the same `+1` padding.

Two tiles cost `2 × BM × BK × 4` bytes for floats. At `32 × 32` that is `2 × 32 × 32 × 4 = 8 KB`. That fits easily in 228 KB. Assignment 1.2 compares naive against this shared-memory tiled kernel, at `4096 × 4096`. The code belongs in that assignment. Today the algorithm is enough.

## Rule for today

Inside the box, GPUs talk on **NVLink + NVSwitch**, not on PCIe. Inside one GPU, a warp should issue a **128-byte aligned, stride-1** global load. Shared memory has **32 banks**. If a tile is 32 wide, pad it. A GEMM tile sits in SRAM so that reuse does not fall back to HBM after the 64 MB matrix no longer fits in L2.
