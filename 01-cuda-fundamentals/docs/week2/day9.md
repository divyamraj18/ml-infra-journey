# Day 9: InfiniBand, RoCE, aur TCP kyun nahi chalega

Day 8 box ke andar khatam hota hai. NVSwitch ka last port chassis pe rukta hai. Do DGX machines ke beech ka traffic **inter-node** hai. Uske liye NIC aur ek alag fabric chahiye.

## TCP/IP wala raasta

GPU ke paas jo gradient pada hai woh HBM mein hai. TCP use socket se bhejta hai, aur socket CPU ki DRAM pe hai. Bina RDMA ke path yeh hai:

```text
GPU0 HBM
  -> cudaMemcpy, PCIe
CPU0 DRAM
  -> kernel TCP stack: copy, checksum, context switch
NIC0
  -> wire
NIC1
  -> kernel TCP
CPU1 DRAM
  -> cudaMemcpy, PCIe
GPU1 HBM
```

Payload do baar PCIe pe chalti hai, aur beech mein CPU kernel ke through. Har chhote message pe syscall hai. Latency tens to hundreds of microseconds. Bandwidth CPU aur PCIe pe atak jaati hai, NIC ke wire rate pe nahi.

LLM training mein yeh har step pe hota hai. Data parallel har step **AllReduce** karta hai: har GPU ke gradients ko sum karke sabko wahi sum wapas dena. 7B model, BF16 gradients = `7e9 × 2 bytes = 14 GB`. Yeh 14 GB har step pe fabric pe move hote hain.

Ring AllReduce ka data volume, baad mein Day 11 pe detail se, lagbhag `2 × S` bytes bhejta hai jab GPUs bahut hon. Ek NIC, 400 Gb/s:

```text
S = 14 GB = 112 Gb
2S / 400 Gb/s ≈ 0.56 s
```

Yeh ek single-rail, overlap ke bina, upper-bound sketch hai. Asli NCCL isse better karta hai kyunki 8 NICs aur compute overlap hota hai. Point yeh hai: TCP is number ko aur bigaad deta hai, kyunki 400 Gb/s tak pahunchta hi nahi. Agar compute step 100 ms ka hai aur network 500 ms ka, to GPU us 500 ms mein khali baitha hai. Wahi Day 2 wala stall, ab HBM ki jagah network.

## InfiniBand

**InfiniBand (IB)** HPC ka fabric hai. Ethernet nahi hai. Link layer **credit-based** hai: sender tabhi packet daalta hai jab receiver ne credits diye hon. Switch buffer full hone se pehle traffic rukta hai. Packet drop design ka hissa nahi hai. TCP ki tarah drop karke retransmit nahi karna padta.

RDMA InfiniBand pe native hai. NIC remote machine ki registered memory mein seedha likhta hai. CPU us payload ko touch nahi karta.

Generations, jo clusters pe dikhengi:

| Gen | Per-port rate | NIC jo aksar saath aata hai |
|---|---|---|
| HDR | 200 Gb/s | ConnectX-6 |
| NDR | 400 Gb/s | ConnectX-7 |
| XDR | 800 Gb/s | agla generation |

DGX H100 typically **8× ConnectX-7**, har ek 400 Gb/s NDR, aur **Quantum-2** InfiniBand switches. 8 NICs isliye kyunki 8 GPUs hain: ek GPU ka ek NIC, ek **rail**. Rail Day 12 pe topology ban jaayega.

Chhote messages pe latency ~1 microsecond ki class mein hai, TCP ke tens of microseconds ke against. Bade gradients pe limit port ki bandwidth hai, syscall nahi.

## RoCE

**RoCE** = RDMA over Converged Ethernet. RDMA ke verbs (queue pair, write, read) wahi hain, lekin wire Ethernet hai.

**RoCE v2** UDP/IP mein wrap hota hai, isliye routers ke across jaa sakta hai. UDP port **4791**. RoCE v1 sirf ek L2 subnet mein tha. Clusters v2 use karte hain.

Ethernet default mein drop kar sakta hai. RDMA drop pasand nahi karta, kyunki uske paas TCP jaisa reliable byte stream nahi hai. Isliye RoCE ko ek managed fabric chahiye:

- **PFC** (Priority Flow Control): ek priority ko pause frame se roko, poore link ko nahi. Loss kam hota hai. Galat config pe **PFC storm** aur head-of-line blocking hota hai. Poora cluster pause pe atak sakta hai.
- **ECN**: switch packet ko drop karne se pehle mark karta hai.
- **DCQCN**: sender ECN dekh kar apni rate ghatata hai. RoCE ka congestion control.

IB yeh sab credit se khud karta hai. RoCE yeh sab switch config se karta hai. Config sahi ho to bandwidth IB ke kareeb hoti hai. Config kharab ho to training "sahi" chalti hai aur step time 2× ho jaata hai. Debug Day 12 pe hai.

## Kaun sa fabric

| | InfiniBand | RoCE v2 |
|---|---|---|
| Wire | Apna IB fabric | Ethernet |
| Loss | Credits, drop nahi | PFC + ECN + DCQCN se bachao |
| RDMA | Native | Same verbs, UDP pe |
| Ops cost | Alag switches, alag tooling | Ethernet jo already hai, lekin PFC nazuk hai |
| Kidhar dikhega | DGX SuperPOD, NVIDIA reference | Bahut saare cloud GPU clusters |

Dono ka data path RDMA hai. TCP unmein se koi nahi hai. Agar `NCCL` log mein `Socket` dikhe, to tum inme se kisi pe nahi ho. Woh Day 12 ka pehla check hai.

## Aaj ka rule

NVSwitch box ke andar. Box ke bahar **InfiniBand ya RoCE**, dono RDMA. TCP GPU gradient ko CPU DRAM aur kernel se guzaarta hai, isliye LLM ke step time mein GPU ko network ka intezaar karna padta hai.
