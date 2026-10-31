# Day 45: Fractional sharing

Bina MIG ke do pods ko same GPU pe time-slice karna: MPS, ya shared annotation. Yeh packing hai.

Isolation kamzor hai. Ek pod dusre ki memory kha sakta hai. Ek kernel dusre ke latency ko hila sakta hai.

MIG isolation deta hai. Fractional sirf "ek GPU pe do chhote kaam" deta hai, jab tum noise accept kar lo.

**Rule:** isolation chahiye to MIG. Sirf packing chahiye to fractional share.
