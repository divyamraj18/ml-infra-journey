# Day 12: NCCL debug aur topology

Collective sahi result de sakta hai aur phir bhi slow ho sakta hai. Galat fabric pe AllReduce ka jawab wahi hota hai. Farq step time mein hota hai. Debug ka kaam yeh pakadna hai ki bytes NVLink aur IB pe gaye, ya TCP socket pe.

## Log on karo

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL
NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log
```

`INFO` algorithm, protocol, aur device likhta hai. `TRACE` usse bahut lamba hai. Pehle INFO.

`SUBSYS` filter hai. `INIT` bootstrap, `NET` kaun sa NIC, `GRAPH` ring/tree, `COLL` collective. Saara log ek stream mein mat lo.

DDP har GPU ke liye alag process hai. Sab stdout pe likhein to lines mix ho jaati hain. `NCCL_DEBUG_FILE` mein `%h` hostname hai, `%p` pid. Ek file, ek process.

## Jo line asli bug hai

Data path ke naam:

| Log mein | Matlab |
|---|---|
| `P2P/NVLink` ya `NVLS` | Intra-node sahi jagah hai. |
| `NET/IB` | InfiniBand. Yahi chahiye IB cluster pe. |
| `NET/RoCE` ya IB HCA jo Ethernet mode mein ho | RoCE. Theek hai agar fabric RoCE hai. |
| `NET/Socket` ya `via Socket` | **TCP fallback.** Training chalegi. Bandwidth PCIe + CPU pe gir jaayegi. |

Socket line ko warning mat samajhna agar sirf **bootstrap** ke liye hai. Bootstrap TCP se QP exchange karta hai. Data collective uske baad `NET/IB` hona chahiye. Agar collective khud Socket pe hai, to NIC use hi nahi ho raha.

Protocol ke naam usi log mein hote hain: **LL**, **LL128**, **Simple**. LL chhote messages. Simple bade. Bada gradient Simple pe hona chahiye.

## Variables jo path badal dete hain

| Variable | Asar |
|---|---|
| `NCCL_IB_DISABLE=1` | IB/RoCE band. Data Socket pe chala jaata hai. Kabhi test ke liye, kabhi koi purani script ise chhod gayi hoti hai. |
| `NCCL_P2P_DISABLE=1` | GPU peer access band. Box ke andar bhi NVLink nahi, traffic host ke through. |
| `NCCL_IB_HCA` | Kaun se ConnectX use hon. Galat naam pe NCCL unhe dhundhta hi nahi, Socket pe gir sakta hai. |
| `NCCL_SOCKET_IFNAME` | Bootstrap ka Ethernet interface. `eth0` vs `ib0`. Data path nahi hai, lekin galat interface pe ranks ek doosre ko discover nahi kar pate. |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA tabhi jab GPU aur NIC itne paas hon. Values: `PIX`, `PXB`, `PHB`, `LOC`, `SYS`. `SYS` matlab CPU sockets ke across bhi allow. Default door wale pair pe GPUDirect band kar sakta hai. Log mein "GPU Direct RDMA disabled" usi ka signal hai. |
| `NCCL_TOPO_DUMP_FILE` | XML. GPU, NIC, NVSwitch, PCIe distance. |
| `NCCL_GRAPH_DUMP_FILE` | Kaun se rings bane. |

Inhe production pe andha copy mat karna. Pehle INFO log dekho, phir ek variable.

## Topology

`nvidia-smi topo -m` ek matrix deta hai. Cell do devices ka distance hai.

| Code | Matlab |
|---|---|
| `NV#` | NVLink, `#` links. Best GPU-GPU. |
| `PIX` | Same PCIe switch. Best GPU-NIC for GPUDirect. |
| `PXB` | PCIe switches ke across, same root complex ke through. |
| `PHB` | Same CPU, alag PCIe host bridge. |
| `NODE` | Same CPU socket, aur door. |
| `SYS` | Doosra CPU socket. QPI/UPI cross. GPUDirect yahan aksar band. |

DGX H100 pe har GPU ka apna ConnectX hota hai, aur woh pair `PIX` hona chahiye. GPU0 ka traffic agar GPU4 wale NIC se bahar jaaye, to intra-node NVLink ke baad bhi ek extra hop aur ek socket cross hai.

**Rail-optimized** fabric: node ke GPU `i` ka NIC, har node pe, same leaf switch pe jaata hai. Us rail ka naam rail `i`. AllReduce ka ring us rail pe reh sakta hai. Spine cross nahi hota. 8 GPU × 8 NIC wale node pe NCCL 8 rings banata hai, ek ring ek rail. Ek ring sirf ek NIC use kare to baaki 7 ports khali hain. Day 13 ka diagram yahi dikhata hai.

## Benchmark

`nccl-tests` ka `all_reduce_perf` wahi operation chalata hai jo DDP chalata hai, training ke bina. Output mein bus bandwidth hoti hai.

Out-of-place AllReduce ki theoretical bus bandwidth, ring pe, lagbhag link rate ke kareeb honi chahiye jab message bada ho. Agar 400 Gb/s NICs hain aur bus bandwidth 10 Gb/s dikhe, to Socket fallback ya galat NIC pe ho. Number yaad karne se pehle log mein `NET/IB` confirm karo. Galat path ka high score meaningful nahi hota.

## Aaj ka rule

`NCCL_DEBUG=INFO` se yeh padho: collective `P2P/NVLink` pe hai ya `NET/IB` pe, aur kahin `Socket` to nahi. `nvidia-smi topo -m` se GPU aur uske NIC ka distance `PIX` hona chahiye. Rail ka matlab GPU `i` hamesha NIC `i` se bahar jaaye, taaki 8 ports ek saath bharen.
