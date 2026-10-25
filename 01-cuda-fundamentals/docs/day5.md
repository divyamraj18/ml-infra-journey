# Day 5: Mixed precision aur gradient checkpointing

Day 3 mein do alag OOM the. Pehli: weights + Adam = 112 GB, batch se nahi ghat ti. Doosri: activations, batch aur sequence se badhti hain. Aaj dono ke ilaaj ka pehla hissa hai. Mixed precision activations ko chhota karti hai aur GPU ko tez. Gradient checkpointing activations ko aur kam save karti hai, thodi extra math ke badle.

## Ek number kitna bada dabba maangta hai

Har weight ek number hai. Dabbe ka size decide karta hai ki number kitna door tak jaa sakta hai, aur kitna bareek ho sakta hai.

| Format | Bytes | Range | Bareeki | Training mein kaam |
|---|---|---|---|---|
| **FP32** | 4 | Bahut bada, FP32 jaisa | Bahut bareek | Adam ki diary, weight ki master copy |
| **FP16** | 2 | Chhota. Sabse bada number lagbhag 65504 | Theek | Purana mixed precision. Chhote gradients 0 ho jaate hain |
| **BF16** | 2 | FP32 jitna bada | FP16 se kam bareek | Aaj ka LLM training. Range bachi rehti hai |
| **FP8** | 1 | Bahut tight | Kam | Hopper / Blackwell. Aur aadhi memory traffic, scaling ke saath |

Range ka matlab: sabse chhota aur sabse bada number jo dabbe mein ghus sake. Bareeki ka matlab: 1.0000001 aur 1.0000002 ko alag pehchaan sakte ho ya nahi.

Thermometer samjho.

- **FP32** almost absolute zero se lekar suraj tak napta hai, kai decimal ke saath. Mehnga hai, 4 bytes.
- **FP16** sirf ek had tak napta hai, lagbhag 65504. Usse bada number phat jaata hai. Bahut chhota number, jaise `0.00001`, dabbe mein ghus hi nahi pata aur **0** ban jaata hai. Isse **underflow** kehte hain. Gradient 0 ho gaya to weight seekhna band kar deta hai.
- **BF16** wahi bada range rakhta hai jo FP32 rakhta hai, lekin decimal kam hote hain. Direction pata rehti hai: weight badhana hai ya ghatana. Saatvi decimal na pata ho, training chalti rehti hai. Isliye LLM BF16 pasand karte hain.
- **FP8** ek aur aadha dabba hai. Bahut chhote gradients aur bahut bade numbers dono ke liye alag FP8 variants hain. Bina scaling ke training toot jaati hai.

## Mixed precision ka matlab

Poora training FP32 mein mat chalao, aur poora FP16 mein bhi mat chalao. Kaam baanto.

- **Matmul** BF16 ya FP16 mein chale. Tensor cores yahi format tez chalate hain. HBM se har number 4 bytes ki jagah 2 bytes aata hai. Day 2 wali bandwidth yahin aadhi ho jaati hai.
- **Adam ki diary** (`m`, `v`) FP32 mein rahe. Chhota update accumulate karna hai, woh chhote dabbe mein gayab ho jaata hai.
- **Weight ki FP32 copy** rahe. Step FP32 mein lagao, phir us copy ko BF16 mein kaat kar agli matmul ko do.

Yeh wahi Day 3 wali table hai:

| Cheez | Bytes |
|---|---|
| Weight jo matmul padhti hai, BF16 | 2 |
| Gradient, BF16 | 2 |
| Weight ki FP32 copy | 4 |
| Adam m, FP32 | 4 |
| Adam v, FP32 | 4 |
| **Total** | **16** |

7B × 16 = 112 GB. Mixed precision yeh 112 GB aadhi nahi karti. `m`, `v`, aur FP32 copy 4 bytes hi rehte hain.

Asli bachaat do jagah hai:

1. **Activations** ab 2 bytes per number hain, 4 nahi. Batch aur sequence wali OOM aadhi ho sakti hai.
2. **Speed.** Tensor cores BF16 multiply karte hain, aur HBM se aadhe bytes travel karte hain.

FP16 mein ek extra khel hai, **loss scaling**. Gradients bahut chhote hain, FP16 unhe 0 bana deta hai. Loss ko pehle 1000 se guna kar do, gradients bade ho kar dabbe mein ghus jaate hain. Weight update se pehle 1000 se wapas bhaag do. BF16 ko yeh aksar nahi chahiye, kyunki uska range FP32 jaisa hai.

FP8 mein matmul ka number 1 byte ka ho jaata hai. Traffic phir aadhi. Diary phir bhi FP32 mein rehti hai. Yeh Day 5 ka idea hai. Scaling ka detail inference week mein wapas aayega.

## Gradient checkpointing

Mixed precision activation ko aadha karti hai. Lambi sequence pe aadha bhi HBM se bahar ho sakta hai. Tab agla ilaaj: **saare beech ke numbers mat rakho.**

Day 3 wala ghar wapas. Do layer the, beech ka number `6` save karna pada tha, kyunki backward ko woh chahiye. Ab 4 layers maan lo. Forward ke baad desk pe 4 sheets hain: `h1 h2 h3 h4`.

Normal training:

```text
forward:   x → h1 → h2 → h3 → h4
desk pe:   h1  h2  h3  h4     sab save
backward:  h4 se h1 tak, save ki hui sheets padh kar
```

Checkpointing. Sirf beech ki sheet rakho, baaki phenk do.

```text
forward:   x → h1 → h2 → h3 → h4
desk pe:   sirf h2

backward layer 4 aur 3:
  h2 se h3 aur h4 dobara nikaalo, phir backward

backward layer 2 aur 1:
  x se h1 aur h2 dobara nikaalo, phir backward
```

Desk pe 4 sheets ki jagah 1 sheet. Forward ka kuch hissa do baar chala. Memory kam, time zyada.

Yeh ilaaj **activation wali OOM** ka hai. Adam ki 112 GB isse nahi ghat ti. Woh lists agle batch ke liye zinda rehni hi hain. Checkpointing unhe nahi chhuta.

## Aaj ka rule

Chhote dabbe (BF16 / FP8) matmul aur activations ke liye. Bada dabba (FP32) Adam ki diary ke liye. Agar activations phir bhi na ghuse, beech ki kuch sheets phenk do aur backward pe woh hissa dobara chala lo.
