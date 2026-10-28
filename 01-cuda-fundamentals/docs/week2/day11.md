# Day 11: NCCL collectives

**NCCL** (NVIDIA Collective Communications Library) runs one operation across GPUs. PyTorch **DDP** and **FSDP**, Megatron tensor parallel, and vLLM tensor parallel all sit on NCCL. You may never call `ncclAllReduce` yourself. The step time is still NCCL's.

Each GPU has a **rank**. **World size** is the number of ranks. Every rank's buffer has the same shape, unless the collective is AllGather or ReduceScatter.

## Five operations

Four ranks. Each rank holds a small vector. The numbers only show the operation.

```text
rank0: [1, 2]
rank1: [3, 4]
rank2: [5, 6]
rank3: [7, 8]
```

**Broadcast.** One root's buffer is copied to everyone. If root is 0, every rank becomes `[1, 2]`. Everyone else's data is overwritten.

**Reduce.** The same index is combined with an op, and the result stays only on the root. Sum, root 0:

```text
rank0: [1+3+5+7, 2+4+6+8] = [16, 20]
rank1, rank2, rank3: their buffers do not hold the reduced result
```

**AllReduce.** Reduce, then the result is on **every** rank. After the sum, all four ranks hold `[16, 20]`. This is the data-parallel gradient step. Each rank computed a gradient on its own batch. After the AllReduce sum, every rank has the full gradient, so the Adam update can run locally. If you want the average, divide by world size after the sum. The framework does that divide. NCCL does the sum.

**AllGather.** Each rank contributes a shard. Everyone receives the concatenation.

```text
rank0 contributes [1]
rank1 contributes [3]
rank2 contributes [5]
rank3 contributes [7]
after: every rank has [1, 3, 5, 7]
```

The output buffer is larger than the input, by a factor of world size.

**ReduceScatter.** Reduce and scatter together. The sum is computed, but each rank keeps only its own piece.

```text
shards of the sum [16, 20]:
rank0 keeps 16
rank1 keeps 20
```

That picture is two elements and two ranks. With four ranks the vector is longer, and each rank keeps `1/N` of the elements.

AllReduce = ReduceScatter + AllGather. NCCL can split it that way internally. FSDP and ZeRO use the split on purpose: they do not need the full gradient on every GPU, only their shard. ReduceScatter is then enough instead of AllReduce, and about half the traffic goes away.

## Which parallelism uses which collective

| Pattern | Collective | Why |
|---|---|---|
| Data parallel, DDP | **AllReduce** on gradients | Every rank needs the full summed gradient, because every rank holds the full model. |
| FSDP / ZeRO-2, ZeRO-3 | **ReduceScatter** on gradients, **AllGather** on parameters | Each rank holds only `1/N` of the state. |
| Tensor parallel, column-parallel linear | **AllGather** | Each GPU produced some columns of the output. The next layer needs the full row. |
| Tensor parallel, row-parallel linear | **AllReduce** or **ReduceScatter** | Each GPU produced a partial sum. Those partials have to be added. |
| MoE expert parallel | **AlltoAll** | A token has to land on the expert the router picked. This is not one of the five above. The core of this plan is ring AllReduce. Remember AlltoAll here. The detail comes later. |

## Ring AllReduce

N ranks in a ring: `0 → 1 → 2 → … → N-1 → 0`. The buffer is split into N chunks.

**ReduceScatter phase**, N−1 steps. On each step every rank sends one chunk to the next rank and adds the chunk it receives into its own chunk. After N−1 steps, each chunk is fully summed, and that sum lives on one rank.

**AllGather phase**, another N−1 steps. Those summed chunks travel around the ring with no further add, until every rank has every chunk.

Bytes each rank sends:

```text
2 * (N - 1) / N * S
```

For large N this is about **`2S`**. S is the buffer size. On 8 GPUs the factor is `2 * 7/8 = 1.75`. On 14 GB of gradients, each GPU puts about `1.75 × 14 ≈ 24.5 GB` onto its link over the whole ring.

That is bandwidth-optimal. The naive pattern is "everyone sends to the root, the root sends the result back." The root's link then carries `(N−1) × S` in and `(N−1) × S` out. The root becomes the bottleneck. In a ring, every link does the same amount of work.

**Tree** AllReduce is for small messages. Depth is `log N`, so latency is lower. Some bytes cross the same links more than once, so a large buffer is worse than a ring. NCCL picks the algorithm from the size. A small tensor uses a tree. A large tensor uses a ring. The Day 12 log prints the algorithm name.

## Reduction inside the switch

**SHARP** does the reduction on an InfiniBand switch. GPUs send partial chunks to the switch. The switch returns the sum. The ring does not have to pass through every GPU.

**NVLS** (NVLink SHARP) is the same idea on NVSwitch, starting with Hopper. An intra-node AllReduce can run on the switch ALU. With 8 GPUs in one box, a lot of gradient traffic finishes on NVLink and never reaches a NIC.

Both work only when NCCL can enable them and the hardware is there. If the log does not say `NVLS` or `SHARP`, the plain ring is running. That is not wrong. The switch shortcut is just not in use.

## Rule for today

DDP uses **AllReduce**. FSDP uses **ReduceScatter + AllGather**. Tensor parallel uses **AllGather** or **AllReduce**, depending on how the layer is cut. On a ring, each GPU sends about **`2 × (N−1)/N × S`** bytes. Small messages use a tree. Large messages use a ring. Inside a Hopper box, NVLS can move that ring onto the switch.
