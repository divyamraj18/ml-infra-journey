# Day 30: Cache layer

Dataset remote hai. Har epoch wahi bytes dobara khinchna mehnga hai. **Alluxio**, ya Ray Data ka local cache, remote object ko node ke disk ya RAM pe rakh leta hai. Agli read local hoti hai.

Cache tab jeetta hai jab data **dohrata** hai, jaise kai epochs. Ek baar stream karke bhool jao to cache sirf extra copy hai. Usse read tez nahi hoti, disk bhar jaata hai.

**Rule:** cache repeated reads ke liye hai. One-shot stream ke liye nahi.
