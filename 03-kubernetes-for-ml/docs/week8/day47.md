# Day 47: Volcano

**Volcano** Kubernetes pe gang aur queue lagata hai. Job ek **PodGroup** hai.

```text
minMember: 8
```

8 pods schedule na hon to 0 chalte hain. 8 milte hi sab start hote hain.

Default scheduler yeh nahi jaanta. Woh pehli khali GPU pe pehla pod rakh deta hai.

**Rule:** Volcano `minMember` se gang enforce karta hai.
