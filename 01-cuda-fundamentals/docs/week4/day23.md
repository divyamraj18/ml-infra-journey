# Day 23: Column parallel aur row parallel

**Column parallel.** Weight **columns** pe cut hai. Har GPU poora `X` padhta hai aur `W` ke kuch columns se output ka tukda banata hai. Agla layer poori row chahe to **AllGather**.

```text
X @ [W0 W1 W2 W3]  ->  [Y0 Y1 Y2 Y3]  -> AllGather -> Y
```

**Row parallel.** Weight **rows** pe cut hai. `X` bhi columns pe cut hona chahiye. Column-parallel ka output naturally cut hota hai, isliye woh agla input ban sakta hai. Har GPU partial `Y` banata hai. Same shape ke partials **AllReduce** ya **ReduceScatter** se judte hain.

```text
[X0 X1 X2 X3] @ rows of W  ->  partial Y on each GPU  -> AllReduce -> Y
```

Megatron ek MLP ko aise jodta hai: pehla linear column-parallel, activation local, doosra linear row-parallel. Beech mein AllGather nahi chahiye, kyunki row-parallel ko split activation chahiye. End pe ek AllReduce. Beech ka collective hat jaata hai.

TP ka message DP ke 14 GB AllReduce se chhota hota hai. Lekin woh **har layer, forward aur backward** pe hota hai. Frequency high. Bandwidth NVLink jaisi honi chahiye.

**Rule:** column-parallel AllGather karta hai. Row-parallel AllReduce karta hai. MLP mein dono ko jod do to beech ka AllGather hat jaata hai.
