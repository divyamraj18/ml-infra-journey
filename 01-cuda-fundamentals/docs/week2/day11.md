# Day 11: NCCL collectives

**NCCL** (NVIDIA Collective Communications Library) woh library hai jo GPUs ke beech ek hi operation ko poora karti hai. PyTorch **DDP** aur **FSDP**, Megatron ka tensor parallel, aur vLLM ka tensor parallel, sab ke neeche NCCL hota hai. Tum `ncclAllReduce` khud na likho, phir bhi step time usi ka hota hai.

Har GPU ka ek **rank** hota hai. `world size` = kitne ranks. Buffer sab pe same shape ka hota hai, jab tak collective AllGather ya ReduceScatter na ho.

## Paanch operations

Chaar ranks. Har rank pe ek chhota vector. Numbers operation dikhane ke liye hain.

```text
rank0: [1, 2]
rank1: [3, 4]
rank2: [5, 6]
rank3: [7, 8]
```

**Broadcast.** Ek root ki copy sabko. Root 0 ho to sab `[1, 2]` pe aa jaate hain. Baaki ka apna data overwrite ho jaata hai.

**Reduce.** Same index pe op lagta hai, result sirf root pe. Sum, root 0:

```text
rank0: [1+3+5+7, 2+4+6+8] = [16, 20]
rank1, rank2, rank3: unka buffer reduce ka result nahi rakhta
```

**AllReduce.** Reduce, phir result **sab** ranks pe. Sum ke baad chaaron pe `[16, 20]`. Data parallel ka gradient step yahi hai. Har rank ne apne batch ka gradient nikala. AllReduce sum ke baad har rank ke paas poora gradient hai, tab Adam update local chal sakta hai. Average chahiye ho to sum ke baad world size se divide hota hai. Woh NCCL ke bahar, framework karta hai. NCCL sum karta hai.

**AllGather.** Har rank ek shard deta hai. Sab ke paas concatenation aati hai.

```text
rank0 contributes [1]
rank1 contributes [3]
rank2 contributes [5]
rank3 contributes [7]
after: every rank has [1, 3, 5, 7]
```

Output buffer input se bada hai, world size guna.

**ReduceScatter.** Reduce bhi, scatter bhi. Sum banta hai, lekin har rank sirf apna hissa rakhta hai.

```text
shards of the sum [16, 20]:
rank0 keeps 16
rank1 keeps 20
```

(Do elements, do ranks wale chhote example mein. Chaar ranks pe vector lamba hota hai aur har rank `1/N` elements rakhta hai.)

AllReduce = ReduceScatter + AllGather. NCCL andar yeh tod sakta hai. FSDP / ZeRO is tod ka fayda uthate hain: unhe poora gradient har GPU pe nahi chahiye, sirf apna shard. Tab AllReduce ki jagah ReduceScatter kaafi hai, aur aadha traffic bachta hai.

## Kaun si parallelism ko kaun sa collective

| Pattern | Collective | Kyun |
|---|---|---|
| Data parallel, DDP | **AllReduce** on gradients | Har rank ko poora summed gradient chahiye, kyunki har rank poora model rakhta hai. |
| FSDP / ZeRO-2, ZeRO-3 | **ReduceScatter** gradients, **AllGather** parameters | Har rank sirf `1/N` state rakhta hai. |
| Tensor parallel, column-parallel linear | **AllGather** | Har GPU ne output ke kuch columns banaye. Agli layer ko poori row chahiye. |
| Tensor parallel, row-parallel linear | **AllReduce** ya **ReduceScatter** | Har GPU ne partial sum banaya. Unhe jodna hai. |
| MoE expert parallel | **AlltoAll** | Token us expert pe jaaye jispe router ne bheja. Yeh paanch mein se nahi hai. Plan ka core Ring AllReduce hai. AlltoAll yahin se yaad rakho, detail baad mein. |

## Ring AllReduce

N ranks ek ring: `0 → 1 → 2 → … → N-1 → 0`. Buffer N chunks mein kata.

**ReduceScatter phase**, N−1 steps. Har step pe har rank ek chunk agle rank ko bhejta hai aur jo chunk aaya use apne chunk mein add kar leta hai. N−1 steps ke baad har chunk kisi ek rank pe fully summed hai.

**AllGather phase**, aur N−1 steps. Woh summed chunks ring pe ghoomte hain, bina add kiye, jab tak sab ke paas sab chunks na hon.

Har rank jitna data bhejta hai:

```text
2 * (N - 1) / N * S
```

N bada ho to yeh **`2S`** ke kareeb hai. S = buffer size. 8 GPUs pe factor `2 * 7/8 = 1.75`. Matlab 14 GB gradients pe har GPU lagbhag `1.75 × 14 ≈ 24.5 GB` apne link pe daalta hai, poori ring ke dauran.

Yeh bandwidth-optimal hai. Naive tarika "sab root ko bhejo, root sabko wapas bheje" hai. Usme root ke link pe `(N−1) × S` aata hai aur wapas jaata hai. Root atak jaata hai. Ring mein har link same kaam karta hai.

**Tree** AllReduce chhote messages ke liye hai. Depth `log N` hai, isliye latency kam hai. Bytes kuch links pe dobara chalte hain, isliye bade buffer pe ring se bura. NCCL size dekh kar algorithm chunta hai. Chhota tensor tree, bada tensor ring. Day 12 ka log algorithm ka naam likhta hai.

## Switch ke andar ka reduce

**SHARP** InfiniBand switch pe reduction hai. GPUs partial chunks switch ko dete hain, switch sum karke result de deta hai. Poora ring GPUs ke through nahi ghuma.

**NVLS** (NVLink SHARP) wahi idea NVSwitch pe, Hopper se. Intra-node AllReduce switch ke ALU pe ho sakta hai. 8 GPUs ek box mein hon to bahut saara gradient traffic NVLink pe hi khatam ho jaata hai, NIC tak jaaye bina.

Dono tab kaam karte hain jab NCCL unhe enable paye aur hardware maujood ho. Log mein `NVLS` ya `SHARP` na dikhe to purana ring chal raha hai. Woh galat nahi, bas switch wala shortcut nahi hai.

## Aaj ka rule

DDP = **AllReduce**. FSDP = **ReduceScatter + AllGather**. Tensor parallel = **AllGather** ya **AllReduce**, layer ke cut pe depend karta hai. Ring pe har GPU lagbhag **`2 × (N−1)/N × S`** bytes bhejta hai. Chhota message tree, bada message ring. Hopper box ke andar NVLS yeh ring switch pe shift kar sakta hai.
