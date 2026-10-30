# Day 11: NCCL collectives

**NCCL** (NVIDIA Collective Communications Library) GPUs ke beech ek operation poora karti hai. PyTorch **DDP** aur **FSDP**, Megatron ka tensor parallel, aur vLLM ka tensor parallel, sab ke neeche NCCL hota hai.

Tum khud `ncclAllReduce` na likho, phir bhi step time usi ka hota hai.

Har GPU ka ek **rank** hota hai. **World size** kitne ranks hain. Jab tak collective AllGather ya ReduceScatter na ho, har rank ka buffer same shape ka hota hai.

## Paanch operations, ek chhoti class

Chaar ranks. Har rank ke paas ek chhota vector. Numbers sirf operation dikhane ke liye hain.

```text
rank0: [1, 2]
rank1: [3, 4]
rank2: [5, 6]
rank3: [7, 8]
```

Socho chaar students ke paas apni copy hai. Teacher ek collective bolta hai. Operation yeh decide karta hai ki baad mein kis ke paas kya bacha.

**Broadcast.** Ek root ki copy sabko milti hai. Root 0 ho to har rank `[1, 2]` ban jaata hai. Baaki ka apna data overwrite ho jaata hai.

**Reduce.** Same index pe op lagta hai. Result sirf root pe rehta hai. Sum, root 0:

```text
rank0: [1+3+5+7, 2+4+6+8] = [16, 20]
rank1, rank2, rank3: inke buffer mein reduced result nahi hai
```

**AllReduce.** Reduce, phir result **har** rank pe. Sum ke baad chaaron ke paas `[16, 20]` hai.

Yeh data parallel ka gradient step hai. Har rank ne apne batch pe gradient nikala. AllReduce sum ke baad har rank ke paas poora gradient hai, tab Adam ka update local chal sakta hai. Average chahiye ho to sum ke baad world size se divide hota hai. Woh divide framework karta hai. NCCL sum karta hai.

**AllGather.** Har rank ek shard deta hai. Sabke paas concatenation aa jaati hai.

```text
rank0 contributes [1]
rank1 contributes [3]
rank2 contributes [5]
rank3 contributes [7]
after: every rank has [1, 3, 5, 7]
```

Output buffer input se bada hai, world size guna.

**ReduceScatter.** Reduce bhi, scatter bhi. Sum banta hai, lekin har rank sirf apna tukda rakhta hai.

```text
sum ke tukde [16, 20]:
rank0 keeps 16
rank1 keeps 20
```

Yeh picture do elements aur do ranks ki hai. Chaar ranks pe vector lamba hota hai, aur har rank `1/N` elements rakhta hai.

AllReduce = ReduceScatter + AllGather. NCCL andar yeh tod sakta hai. FSDP aur ZeRO yeh tod jaanbujh kar use karte hain. Unhe poora gradient har GPU pe nahi chahiye, sirf apna shard. Tab AllReduce ki jagah ReduceScatter kaafi hai, aur lagbhag aadha traffic bachta hai.

## Kaun si parallelism, kaun sa collective

| Pattern | Collective | Kyun |
|---|---|---|
| Data parallel, DDP | Gradients pe **AllReduce** | Har rank ko poora summed gradient chahiye, kyunki har rank poora model rakhta hai. |
| FSDP / ZeRO-2, ZeRO-3 | Gradients pe **ReduceScatter**, parameters pe **AllGather** | Har rank sirf `1/N` state rakhta hai. |
| Tensor parallel, column-parallel linear | **AllGather** | Har GPU ne output ke kuch columns banaye. Agli layer ko poori row chahiye. |
| Tensor parallel, row-parallel linear | **AllReduce** ya **ReduceScatter** | Har GPU ne partial sum banaya. Unhe jodna hai. |
| MoE expert parallel | **AlltoAll** | Token us expert pe jaaye jispe router ne bheja. Yeh upar ke paanch mein se nahi hai. Is plan ka core ring AllReduce hai. AlltoAll yahin yaad rakho. Detail baad mein. |

## Ring AllReduce: relay race

N ranks ek ring mein: `0 → 1 → 2 → … → N-1 → 0`. Buffer N chunks mein kata hua hai.

Socho ek baton jo circle mein ghoomti hai. Farq yeh hai ki beech mein log apna hissa jodte jaate hain, phir joda hua baton sabko dikhaya jaata hai.

**ReduceScatter phase.** N−1 steps. Har step pe har rank ek chunk agle rank ko bhejta hai, aur jo chunk aaya use apne chunk mein add kar leta hai. N−1 steps ke baad har chunk kisi ek rank pe poora summed hai.

**AllGather phase.** Aur N−1 steps. Woh summed chunks ring pe ghoomte hain, ab add nahi hota, jab tak har rank ke paas har chunk na ho.

Har rank jitna data bhejta hai:

```text
2 * (N - 1) / N * S
```

N bada ho to yeh **`2S`** ke kareeb hai. S buffer ka size hai. 8 GPUs pe factor `2 × 7/8 = 1.75` hai. 14 GB gradients pe har GPU poori ring ke dauran lagbhag `1.75 × 14 ≈ 24.5 GB` apni link pe daalta hai.

Yeh bandwidth-optimal hai. Naive tarika: sab root ko bhejo, root sabko wapas bheje. Root ki link pe `(N−1) × S` aata hai aur utna hi wapas jaata hai. Root atak jaata hai. Ring mein har link same kaam karti hai.

**Tree** AllReduce chhote messages ke liye hai. Depth `log N` hai, isliye latency kam hai. Kuch bytes same links pe dobara chalte hain, isliye bada buffer ring se bura hai. NCCL size dekh kar algorithm chunta hai. Chhota tensor tree. Bada tensor ring. Day 12 ka log algorithm ka naam likhta hai.

## Switch ke andar ka add

**SHARP** InfiniBand switch pe reduction hai. GPUs partial chunks switch ko dete hain. Switch sum karke result de deta hai. Poora ring har GPU se nahi guzarta.

**NVLS** (NVLink SHARP) wahi idea NVSwitch pe hai, Hopper se. Intra-node AllReduce switch ke ALU pe ho sakta hai. 8 GPUs ek box mein hon to bahut saara gradient traffic NVLink pe hi khatam ho jaata hai. NIC tak jaaye bina.

Dono tab chalte hain jab NCCL unhe enable kar paaye aur hardware maujood ho. Log mein `NVLS` ya `SHARP` na dikhe to purana ring chal raha hai. Woh galat nahi hai. Switch wala shortcut use nahi ho raha.

## Aaj ka rule

DDP = **AllReduce**. FSDP = **ReduceScatter + AllGather**. Tensor parallel = **AllGather** ya **AllReduce**, layer ka cut decide karta hai. Ring pe har GPU lagbhag **`2 × (N−1)/N × S`** bytes bhejta hai. Chhota message tree. Bada message ring. Hopper box ke andar NVLS us ring ko switch pe shift kar sakta hai.
