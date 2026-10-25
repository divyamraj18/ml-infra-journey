# Day 3: Training OOM — HBM kaun bhar deta hai

Day 2 mein HBM GPU ka bada godam thi. Bada, lekin limited. H100 pe lagbhag 80 GB. Training ke dauraan kai lists ek saath isi godam mein rehti hain. OOM ka matlab GPU kharab hona nahi hai. Matlab in lists ka sum 80 GB se bada ho gaya.

Yeh lists disk pe save ki hui files nahi hain. Yeh training ke ek step ke beech GPU ki memory mein zinda rehti hain.

## Pehle ek chhota model, taaki buckets dikhein

Socho model sirf itna hai: ghar ki keemat = kamron ki sankhya × ek weight.

- Kamre = 3
- Weight = 2
- Model ki prediction = 3 × 2 = 6 lakh
- Asli keemat = 10 lakh
- Galati = 4 lakh kam

Is chhote model mein weight ek hi number hai. 7B model mein aise numbers 7 billion hain. Idea wahi rehta hai.

## Weights kya hain

Weight woh number hai jo model ne seekha hai. Upar wale example mein weight `2` hai, matlab "ek kamre ki keemat 2 lakh."

7B model mein yeh ek lambi list hai, 7 billion numbers. BF16 mein ek number 2 bytes ka hota hai.

7 billion × 2 bytes = 14 GB. Sirf weights. Training ka step shuru hone se pehle yeh list HBM mein baith chuki hoti hai, aur agle step mein bhi wahi rehti hai. Seekh kar weight badalta hai, list delete nahi hoti.

## Gradient kyun rakhte hain

Prediction 6 lakh hai, asli keemat 10 lakh. Weight ko badhana hai ya ghatana?

**Gradient woh number hai jo ek weight ko kehta hai: upar jao ya neeche, aur kitna.**

Is example mein gradient kehta hai weight `2` ko badhao, kyunki prediction kam pad gayi. Har weight ka apna gradient hota hai. 7 billion weights hon to 7 billion gradients hote hain. List ka shape weights jaisa hi hai.

Gradient ke bina training andheri hai. Galati pata hai, lekin yeh nahi pata ki kaun sa weight us galati ke liye zimmedar hai aur use kis taraf dhakelna hai.

Yeh list step ke beech banti hai, jab backward pass har weight ki zimmedari nikalta hai. Step khatam hote hi gradients pheke jaa sakte hain. Agle batch ke gradients naye hote hain. Peak memory us waqt hoti hai jab yeh list abhi zinda ho.

7B, BF16 gradients = aur 14 GB.

Ek layer pe yeh dikhta kaise hai. Chhoti table, 2 × 2 weights:

```text
weights          gradients
[ 2.0  0.5 ]     [ +0.4  -0.1 ]
[ 1.0  3.0 ]     [ -0.2  +0.7 ]
```

Dono tables same size ki hain. Left wala number model hai. Right wala number usi weight ka nudge hai. `+0.4` ka matlab pehla weight thoda badhe. `-0.1` ka matlab doosra weight thoda ghate.

## Optimizer state kyun rakhte hain

Sirf is ek ghar ke gradient pe weight ko chhod do, to ek ajeeb ghar model ko jhatke se door phenk sakta hai. Agli baar koi mehenga ghar aaya, gradient bahut bada nikla, weight 2 se seedha 50 ho gaya. Model paagal ho jaata hai.

Adam ek diary rakhta hai, taaki ek jhatka poori diary ko na bigade. Har weight ki apni diary hai. Usme do numbers hain:

- **m** — is weight ke pichle gradients ka average. Ek ajeeb ghar average ko thoda hilata hai, poora nahi.
- **v** — is weight ke gradients kitne uchhal rahe hain. Zyada uchhal hon to step chhota lo. Shant hon to step bada lo.

Agle ghar pe Adam poora gradient nahi maanta. Woh m aur v dekh kar ek calm step leta hai.

`m` aur `v` agle batch ke liye zinda rehte hain. Unhe har step pe phenk nahi sakte, kyunki diary ka poora point pichli memory hai. Isliye yeh HBM mein hamesha baithi rehti hai, weights ki tarah.

Dono lists weights jitni lambi hain, aur FP32 mein rehti hain. FP32 ek number 4 bytes ka hai. Chhote updates FP16 ke chhote dabbe mein fit nahi hote, isliye weight ki ek FP32 copy bhi rakhi jaati hai.

Wahi 2 × 2 layer, Adam ke saath, ek saath char tables:

```text
weights BF16       gradients BF16
[ 2.0  0.5 ]       [ +0.4  -0.1 ]
[ 1.0  3.0 ]       [ -0.2  +0.7 ]

Adam m             Adam v
[ 0.3  -0.05 ]     [ 0.02  0.01 ]
[ -0.1  0.4 ]      [ 0.01  0.08 ]
```

Plus weights ki FP32 copy, jo dikhne mein weights jaisi hi table hai, bas har number zyada precision ke saath.

Adam mixed precision ka common hisaab, **har parameter pe lagbhag 16 bytes**. Activations is table mein nahi hain:

| Cheez | Bytes | Step ke baad |
|---|---|---|
| Weight, BF16 | 2 | Rehti hai |
| Gradient, BF16 | 2 | Phenki jaa sakti hai |
| Weight ki FP32 copy | 4 | Rehti hai |
| Adam m | 4 | Rehti hai |
| Adam v | 4 | Rehti hai |
| **Total** | **16** | |

7B × 16 bytes = 112 GB.

Yeh 112 GB activations se pehle hai. 80 GB wali H100 pe 7B ka full Adam fit nahi hota. Batch 1 kar do, phir bhi yeh 112 GB nahi ghat ta, kyunki yeh list model ke size ki hai, batch ke size ki nahi.

## Activation kyun rakhte hain

Ab model do layer ka hai.

- Kamre = 3
- Pehli layer: 3 × 2 = **6**. Yeh beech ka jawab hai.
- Doosri layer: 6 × 1.5 = 9. Yeh final prediction hai.

**6 ko activation kehte hain.** Woh final jawab nahi hai. Woh beech ka number hai jo agle layer ko diya gaya.

Wapas jaate waqt pehli weight (`2`) ko theek karna hai. Uske liye yeh beech wala 6 chahiye, kyunki doosri layer ne 6 ko use karke hi galati paida ki. Agar forward ke turant baad 6 phenk diya, to backward pehli layer tak pahunch hi nahi sakta. Ya to 6 save karo, ya pehli layer dobara chalao.

Training save karna pasand karti hai, kyunki dobara chalana mehnga hai. Isliye forward ke beech ke numbers backward khatam hone tak HBM mein rehte hain. Backward khatam, activations phenk do. Agle batch ke activations naye hote hain.

Activation ki shape weights jaisi nahi hoti. Weights ek layer ki table hain. Activation us batch ke har token ka beech ka jawab hai.

Do sentences, har sentence 3 tokens, aur beech ka jawab 4 numbers lamba ho:

```text
activation shape = [2 sentences, 3 tokens, 4 numbers]

sentence 1
  token "ghar"   [ 0.2  1.0  0.0  0.5 ]
  token "bada"   [ 0.1  0.4  0.8  0.2 ]
  token "hai"    [ 0.0  0.3  0.1  0.9 ]

sentence 2
  token "yeh"    [ 0.6  0.1  0.2  0.0 ]
  token "sasta"  [ 0.2  0.7  0.4  0.1 ]
  token "hai"    [ 0.3  0.0  0.5  0.8 ]
```

Weights is example mein 4 × 4 = 16 numbers ho sakte hain. Activation 2 × 3 × 4 = 24 numbers hai. Batch ya sentence lamba karo, activation badhegi. Weights nahi badhenge.

Asli layer pe hidden size 4096 hota hai, sequence 1024 tokens, batch 2:

2 × 1024 × 4096 = lagbhag 8 million numbers, **sirf ek layer ka ek output**. Model mein aisi layers 30-40 hoti hain, aur har layer ke andar kai beech ke numbers save hote hain.

Attention ka score matrix aur bhi bura hai: batch × heads × sequence × sequence. Sequence double karo, yeh matrix chaar guna ho jaata hai. Lambi context pe OOM aksar yahin se aata hai.

## Do alag OOMs

**Model bada hai.** Weights + gradients + Adam = 112 GB. Batch 1 se yeh nahi ghat ta. 80 GB se pehle hi bahar.

**Batch ya context bada hai.** Model states ghus gaye, lekin sentences lambi karte hi activation ki `[batch, tokens, hidden]` list HBM bhar deti hai.

16 bytes wala number sirf pehli OOM ke liye hai. Activations uske upar alag se baithte hain, aur unka size batch aur sequence se aata hai.
