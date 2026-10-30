# Day 12: NCCL debug aur topology

Collective sahi jawab de sakta hai aur phir bhi slow ho sakta hai. Galat sadak pe AllReduce ka result wahi rehta hai. Farq step time mein dikhta hai.

Debug ka kaam yeh pakadna hai: bytes NVLink aur InfiniBand pe gaye, ya TCP socket pe?

Socho parcel pe ek shipping label hai. Label galat ho to parcel pahunch jaata hai, lekin local post se, express highway se nahi.

## Log on karo

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL
NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log
```

`INFO` algorithm, protocol, aur device likhta hai. `TRACE` usse bahut lamba hai. Pehle INFO.

`SUBSYS` filter hai.

| Subsystem | Kya dikhata hai |
|---|---|
| `INIT` | Bootstrap. Ranks ne ek doosre ko kaise dhunda. |
| `NET` | Kaun sa NIC. |
| `GRAPH` | Ring ya tree. |
| `COLL` | Collective khud. |

DDP har GPU ke liye alag process hai. Sab stdout pe likhein to lines ek doosre mein ghus jaati hain. `NCCL_DEBUG_FILE` mein `%h` hostname hai, `%p` pid. Ek file, ek process.

## Jo line asli bug hai

Data path ke naam:

| Log mein | Matlab |
|---|---|
| `P2P/NVLink` ya `NVLS` | Box ke andar sahi jagah. |
| `NET/IB` | InfiniBand. IB cluster pe yahi chahiye. |
| `NET/RoCE`, ya IB HCA jo Ethernet mode mein ho | RoCE. Theek hai agar fabric RoCE hai. |
| `NET/Socket` ya `via Socket` | **TCP fallback.** Training chalegi. Bandwidth PCIe aur CPU pe gir jaayegi. |

`Socket` ko turant bug mat samajhna agar woh sirf **bootstrap** hai. Bootstrap TCP se queue pairs exchange karta hai. Data collective uske baad `NET/IB` hona chahiye. Collective khud Socket pe hai to NIC data path mein hai hi nahi.

Usi log mein protocol ke naam hote hain: **LL**, **LL128**, **Simple**. LL chhote messages. Simple bade. Bada gradient Simple pe hona chahiye.

## Variables jo sadak badal dete hain

| Variable | Asar |
|---|---|
| `NCCL_IB_DISABLE=1` | IB/RoCE band. Data Socket pe chala jaata hai. Kabhi test ke liye hota hai. Kabhi purani script ise chhod gayi hoti hai. |
| `NCCL_P2P_DISABLE=1` | GPU peer access band. Box ke andar bhi NVLink nahi. Traffic host se guzarta hai. |
| `NCCL_IB_HCA` | Kaun se ConnectX use hon. Galat naam pe NCCL unhe dhundhta hi nahi, aur Socket pe gir sakta hai. |
| `NCCL_SOCKET_IFNAME` | Bootstrap ka Ethernet interface. `eth0` vs `ib0`. Yeh data path nahi hai. Galat interface pe ranks ek doosre ko discover hi nahi kar paate. |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA tabhi jab GPU aur NIC kaafi paas hon. Values: `PIX`, `PXB`, `PHB`, `LOC`, `SYS`. `SYS` CPU sockets ke across bhi allow karta hai. Default door wale pair pe GPUDirect band kar sakta hai. Log line "GPU Direct RDMA disabled" usi ka signal hai. |
| `NCCL_TOPO_DUMP_FILE` | XML. GPU, NIC, NVSwitch, PCIe distance. |
| `NCCL_GRAPH_DUMP_FILE` | Kaun se rings bane. |

Inhe production pe andha copy mat karna. Pehle INFO log padho. Phir ek variable badlo.

## Topology: devices kitne door hain

`nvidia-smi topo -m` ek matrix deta hai. Cell do devices ka distance hai.

| Code | Matlab |
|---|---|
| `NV#` | NVLink, `#` links. GPU se GPU, best. |
| `PIX` | Same PCIe switch. GPU se NIC, GPUDirect ke liye best. |
| `PXB` | PCIe switches ke across, same root complex se. |
| `PHB` | Same CPU, alag PCIe host bridge. |
| `NODE` | Same CPU socket, aur door. |
| `SYS` | Doosra CPU socket. QPI ya UPI cross. GPUDirect yahan aksar band. |

DGX H100 pe har GPU ka apna ConnectX hota hai. Woh pair `PIX` hona chahiye. GPU0 ka traffic agar GPU4 wale NIC se bahar jaaye, to NVLink ke baad bhi ek extra hop aur ek socket cross padta hai.

**Rail-optimized** fabric: har node pe GPU `i` ka NIC same leaf switch pe jaata hai. Us path ka naam rail `i` hai. AllReduce ka ring us rail pe reh sakta hai. Spine cross nahi hota.

8 GPU × 8 NIC wale node pe NCCL 8 rings banata hai, ek ring ek rail. Ek ring sirf ek NIC use kare to baaki 7 ports khali reh jaate hain. Day 13 ka diagram yahi hai.

## Benchmark

`nccl-tests` ka `all_reduce_perf` wahi operation chalata hai jo DDP chalata hai, training loop ke bina. Output mein **bus bandwidth** hoti hai.

Bada out-of-place ring AllReduce, link rate ke kareeb bus bandwidth dikhana chahiye. NICs 400 Gb/s hon aur bus bandwidth 10 Gb/s dikhe, to Socket fallback hai ya galat NIC. Number yaad karne se pehle log mein `NET/IB` confirm karo. Galat path ka bada score matlab nahi rakhta.

## Aaj ka rule

`NCCL_DEBUG=INFO` se yeh padho: collective `P2P/NVLink` pe hai ya `NET/IB` pe, aur `Socket` pe nahi. `nvidia-smi topo -m` pe GPU aur uske NIC ke beech `PIX` hona chahiye. Rail ka matlab GPU `i` hamesha NIC `i` se bahar jaaye, taaki 8 ports ek saath bharen.
