# Day 3: Training OOM — HBM kaun bhar deta hai

Day 2 mein HBM GPU ka bada godam thi. Bada, lekin limited. H100 pe lagbhag 80 GB. Training mein yeh 80 GB teen-chaar buckets milkar bhar dete hain. OOM ka matlab yeh nahi ki GPU kharab hai. Matlab in buckets ka sum HBM se bada ho gaya.

## 1. Model weights

Yeh model khud hai. Har parameter ek number hai.

BF16 ya FP16 mein ek number 2 bytes ka hota hai.

7 billion parameters × 2 bytes = 14 GB. Sirf weights. Calculation shuru hone se pehle yeh HBM mein baith chuke hote hain.

## 2. Gradients

Training mein har weight ka ek gradient chahiye, taaki pata chale weight ko kis direction mein hilana hai. Gradients ka size weights jaisa hota hai.

7B model, BF16 gradients = aur 14 GB.

## 3. Optimizer states

LLM training mein Adam use hota hai. Adam har weight ke liye do extra arrays rakhta hai:

- **m** — ab tak ke gradients ka average
- **v** — ab tak ke squared gradients ka average

Yeh dono FP32 mein rehte hain, 4 bytes per number. Mixed precision mein weights ki ek FP32 copy bhi rakhi jaati hai, kyunki chhote updates FP16 mein vanish ho jaate hain.

Adam mixed precision ka common hisaab, **har parameter pe lagbhag 16 bytes**:

| Cheez | Bytes |
|---|---|
| Weight, BF16 | 2 |
| Gradient, BF16 | 2 |
| Weight ki FP32 copy | 4 |
| Adam m, FP32 | 4 |
| Adam v, FP32 | 4 |
| **Total** | **16** |

7B × 16 bytes = 112 GB.

Yeh sirf weights, gradients, aur Adam ki state hai. Activations abhi include nahi hue. 80 GB wali H100 pe 7B ka full Adam fit nahi hota. OOM yahin se aata hai.

SGD jaise simple optimizer ke states chhote hote hain. LLM Adam use karte hain, isliye 16-byte wala number yaad rakhna hai.

## 4. Activations

Forward pass ke beech ke outputs backward ke liye save hote hain. Inhe activations kehte hain.

Weights batch badalne se nahi badhte. Activations badhte hain, kyunki har example aur har token ka apna beech-ka-hisaab save hota hai.

Do cheezein inhe bada karte hain:

- **Batch size** — ek saath jitne examples
- **Sequence length** — ek example kitne tokens lamba hai

Sabse bhari cheez attention ka score matrix hai: batch × heads × sequence × sequence. Sequence double karo, yeh matrix chaar guna ho jaata hai. Lambi context pe OOM aksar yahin se aata hai.

Layer ke outputs alag se bhi save hote hain: batch × sequence × hidden size × layers. Yeh sequence ke saath seedha badhte hain, square nahi.

FlashAttention baad mein aayega. Woh attention ka poora sequence × sequence matrix save kiye bina kaam karta hai. Uske baad bhi layer outputs reh jaate hain, aur bada batch ya lambi sequence unhe phir bhi badha deti hai.

## Do alag OOMs

**Model bada hai.** Weights + gradients + Adam 80 GB cross kar jaate hain. Batch 1 kar do, phir bhi nahi ghusta. 7B Adam = 112 GB, activations se pehle.

**Batch ya context bada hai.** Model ghusta hai, lekin sequence lambi karte hi activations HBM bhar dete hain.

Pehla OOM model size ka hai. Doosra OOM batch aur sequence ka hai. Dono ka ilaaj alag hai. Ilaaj baad ke dinon mein hai: mixed precision aur gradient checkpointing Day 5 pe, optimizer ko kai GPUs pe baantna Week 4 pe. Aaj sirf buckets pehchaanne hain.
