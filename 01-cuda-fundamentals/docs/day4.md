# Day 4: Execution model aur coalesced access

Day 1 mein thread, block, aur grid ban chuke hain. Aaj yeh hai ki CPU us kaam ko GPU ko kaise thamaata hai, aur ek warp memory ko kaise padhta hai. Nayi cheez **coalesced access** hai. Matlab warp ke 32 threads paas-paas ke addresses padhein, taaki HBM ek trip mein sabko data de de.

## Kernel kya hai

**Kernel** woh kaam hai jo GPU pe chalta hai. CPU use launch karta hai, phir GPU hazaaron threads mein baant kar chalaata hai.

CPU khud woh hazaaron additions nahi karta. CPU sirf yeh kehta hai: "yeh kaam, itni teams, har team mein itne log." Uske baad GPU schedule karta hai.

## Launch se lekar SM tak

12 numbers add karne hain. Tumne decide kiya: har team mein 4 threads, to 3 blocks. Yeh software ka faisla hai, Day 1 wala.

Ab execution:

1. CPU kernel launch karta hai. Saath mein grid ka size jaata hai: 3 blocks, har block mein 4 threads.
2. GPU har block ko kisi khali SM pe bithaata hai. Block 0 ek floor pe, block 1 doosre floor pe. Teesra floor khali ho to block 2 bhi saath chal sakta hai.
3. SM us block ke threads ko **32 ke warp** mein chalaata hai. Yahan team mein sirf 4 log hain, to ek warp banta hai aur 28 seats khali reh jaati hain.
4. Agar koi warp HBM ka intezaar kar raha ho, SM doosra ready warp chala deta hai. Intezar chhupane ka yahi tarika hai.

```text
CPU launch
  grid = 3 blocks
    block 0 (4 threads) --> SM 0 --> warp, 28 seats khali
    block 1 (4 threads) --> SM 1 --> warp, 28 seats khali
    block 2 (4 threads) --> SM 2 --> warp, 28 seats khali
```

Asli kaam mein block mein 128 ya 256 threads hote hain, kyunki 32 ka multiple seats barbaad nahi karta. 256 threads = 8 warps. SM un warps ko baari baari se chalaata hai.

## Coalesced access

HBM se ek number mangwana mehnga hai. Hardware ek number ke liye poori shelf uthata hai, lagbhag **128 bytes** ek trip mein. Ek float 4 bytes ka hai. 128 / 4 = 32 floats. Ek warp mein bhi 32 threads hote hain. Hisaab isliye baithta hai.

**Coalesced:** warp ke 32 threads lagataar 32 floats maangte hain. Ek trip, 128 bytes, sab use ho jaate hain.

**Uncoalesced:** har thread door-door ka float maangta hai. Har trip phir bhi 128 bytes laati hai, lekin usme se sirf 4 bytes kaam aate hain. Baaki 124 bytes barbaad. Bandwidth Day 2 wali, aur bhi bura waste.

Socho librarian. 32 logon ko ek-ek page chahiye.

- Pages 1, 2, 3, ... 32 chahiye. Librarian ek bundle utha kar de deta hai. Ek trip.
- Page 1, page 500, page 10000 chahiye. Librarian 32 alag trips karta hai, aur har trip mein aas-paas ke pages bhi uth kar laata hai jo kisi ne nahi maange.

### Chhota array

8 floats ek line mein pade hain. Warp ko 4 threads maan ke dekho, taaki picture chhoti rahe. Asli warp 32 ka hota hai.

```text
memory:  [ a0  a1  a2  a3  a4  a5  a6  a7 ]
```

Achha padhna. Thread i, index i padhta hai.

```text
t0 → a0
t1 → a1
t2 → a2
t3 → a3
```

Ye chaaron numbers ek doosre se chipke hain. Ek trip.

Bura padhna. Thread i, index `i × 2` padhta hai.

```text
t0 → a0
t1 → a2
t2 → a4
t3 → a6
```

Beech ke a1, a3, a5, a7 bhi trip mein aa sakte hain, lekin in char threads ne unhe maanga hi nahi. Bandwidth aadhi barbaad.

Aur bura: har thread `i × 32` padhe. Tab har thread alag 128-byte shelf maangta hai. 32 threads, 32 trips, har trip se ek float.

### Matrix pe yahi farq

Matrix memory mein **row ke row** store hoti hai. Ek row ke numbers chipke hote hain. Agli row door shuru hoti hai.

```text
row 0:  a00 a01 a02 a03
row 1:  a10 a11 a12 a13
row 2:  a20 a21 a22 a23
row 3:  a30 a31 a32 a33
```

Memory ki asli line yeh hai: `a00 a01 a02 a03 a10 a11 a12 a13 ...`

Ek warp ek **row** padhe: `a00 a01 a02 a03`. Chipke hue. Coalesced.

Ek warp ek **column** padhe: `a00`, phir `a10`, phir `a20`, phir `a30`. Har number ke beech mein poori row ka gap hai. Uncoalesced.

Element-wise add tab tez rehta hai jab thread i, element i padhe. Woh already memory-bound hai. Uncoalesced bana do to wahi kaam aur dheema ho jaata hai, kyunki har byte ke liye aur zyada trips lagti hain.

## Aaj ka rule

Grid, block, thread tum launch karte ho. GPU unhe SM aur warp pe chalaata hai. Warp memory tab sasta padhta hai jab 32 threads ek ke baad ek address padhein. Ek trip, 128 bytes, sab kaam aa jayein.
