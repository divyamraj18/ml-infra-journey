# Day 22: Tensor parallel aur online softmax

Data parallel poora model copy karta hai. Pipeline layers ke groups kaat ta hai. **Tensor parallel (TP)** ek hi matrix ko kaat ta hai. Megatron-LM yahi karta hai.

`Y = X @ W`. `W` ek GPU pe nahi aata, ya aata hai lekin tum use kai GPUs pe faila kar tez karna chahte ho.

Socho ek lamba hisaab chaar logon mein bata. Har koi columns ka tukda karta hai, phir results chipakte ya judte hain. Yeh jodna **har layer** pe hota hai, step ke end pe ek baar nahi. Isliye TP ko **NVLink** chahiye. Box ke bahar InfiniBand pe yeh chatty traffic mehnga hai.

**Online softmax**, FlashAttention ka core idea. Normal attention `seq × seq` score matrix HBM mein likh deta hai. Day 3: sequence double, yeh matrix 4×. FlashAttention scores ko SRAM mein rakhta hai aur HBM mein poori matrix nahi likhta.

Online softmax ek pass mein **max** aur **sum** dono rakhta hai. Naya element purane max se bada ho to purani sum ko rescale karo. Do kernels (pehli max, phir exp) ki jagah ek fused kernel. HBM trips kam. PyTorch `torch.softmax` se compare karna baad ka assignment hai. Code abhi nahi.

**Rule:** TP ek layer kaat ta hai aur har layer pe NCCL chalaata hai. Online softmax `seq × seq` matrix ko HBM se hataata hai.
