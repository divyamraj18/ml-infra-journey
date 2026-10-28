# Day 12: NCCL debug and topology

A collective can return the right answer and still be slow. AllReduce on the wrong fabric produces the same numbers. The difference shows up in step time. Debugging means finding out whether the bytes went over NVLink and InfiniBand, or over a TCP socket.

## Turn the log on

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL
NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log
```

`INFO` prints the algorithm, the protocol, and the device. `TRACE` is much longer. Start with INFO.

`SUBSYS` is a filter. `INIT` is bootstrap. `NET` is which NIC. `GRAPH` is the ring or tree. `COLL` is the collective. Do not take the whole log in one stream.

DDP is one process per GPU. If every process writes stdout, the lines mix. In `NCCL_DEBUG_FILE`, `%h` is the hostname and `%p` is the pid. One file per process.

## The line that is the real bug

Names of the data path:

| In the log | Meaning |
|---|---|
| `P2P/NVLink` or `NVLS` | Intra-node traffic is on the right path. |
| `NET/IB` | InfiniBand. This is what you want on an IB cluster. |
| `NET/RoCE`, or an IB HCA running in Ethernet mode | RoCE. Fine if the fabric is RoCE. |
| `NET/Socket` or `via Socket` | **TCP fallback.** Training still runs. Bandwidth falls onto PCIe and the CPU. |

Do not treat a Socket line as a warning if it is only **bootstrap**. Bootstrap uses TCP to exchange queue pairs. The data collective after that should be `NET/IB`. If the collective itself is on Socket, the NIC is not in the data path.

The same log names the protocol: **LL**, **LL128**, **Simple**. LL is for small messages. Simple is for large ones. A large gradient should be on Simple.

## Variables that change the path

| Variable | Effect |
|---|---|
| `NCCL_IB_DISABLE=1` | Turns IB/RoCE off. Data falls onto Socket. Sometimes this is a test. Sometimes an old script left it set. |
| `NCCL_P2P_DISABLE=1` | Turns off GPU peer access. Even inside the box there is no NVLink. Traffic goes through the host. |
| `NCCL_IB_HCA` | Which ConnectX devices to use. A wrong name means NCCL never finds them, and it can fall back to Socket. |
| `NCCL_SOCKET_IFNAME` | Ethernet interface for bootstrap. `eth0` versus `ib0`. This is not the data path, but the wrong interface means ranks never discover each other. |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA only when the GPU and NIC are close enough. Values: `PIX`, `PXB`, `PHB`, `LOC`, `SYS`. `SYS` also allows pairs that sit across CPU sockets. The default can disable GPUDirect for distant pairs. The log line "GPU Direct RDMA disabled" is that signal. |
| `NCCL_TOPO_DUMP_FILE` | XML of GPUs, NICs, NVSwitch, and PCIe distance. |
| `NCCL_GRAPH_DUMP_FILE` | Which rings were built. |

Do not copy these into production blindly. Read the INFO log first, then change one variable.

## Topology

`nvidia-smi topo -m` prints a matrix. A cell is the distance between two devices.

| Code | Meaning |
|---|---|
| `NV#` | NVLink, `#` links. Best GPU-to-GPU path. |
| `PIX` | Same PCIe switch. Best GPU-to-NIC path for GPUDirect. |
| `PXB` | Across PCIe switches, through the same root complex. |
| `PHB` | Same CPU, different PCIe host bridge. |
| `NODE` | Same CPU socket, farther away. |
| `SYS` | The other CPU socket. Crosses QPI or UPI. GPUDirect is often disabled here. |

On a DGX H100 each GPU has its own ConnectX, and that pair should be `PIX`. If GPU0's traffic leaves through the NIC next to GPU4, you still pay an extra hop and a socket cross after the intra-node NVLink.

A **rail-optimized** fabric sends GPU `i`'s NIC, on every node, to the same leaf switch. That path is rail `i`. An AllReduce ring can stay on that rail and avoid the spine. On a node with 8 GPUs and 8 NICs, NCCL builds 8 rings, one ring per rail. One ring that uses only one NIC leaves the other 7 ports idle. The Day 13 diagram is that picture.

## Benchmark

`all_reduce_perf` from `nccl-tests` runs the same operation DDP runs, without a training loop. The output includes bus bandwidth.

For a large out-of-place ring AllReduce, bus bandwidth should land near the link rate. If the NICs are 400 Gb/s and bus bandwidth prints 10 Gb/s, you are on a Socket fallback or the wrong NIC. Confirm `NET/IB` in the log before you memorize the number. A high score on the wrong path does not mean anything.

## Rule for today

Read `NCCL_DEBUG=INFO` for this: the collective is on `P2P/NVLink` or `NET/IB`, and it is not on `Socket`. `nvidia-smi topo -m` should show `PIX` between a GPU and its NIC. A rail means GPU `i` always exits through NIC `i`, so all 8 ports fill together.
