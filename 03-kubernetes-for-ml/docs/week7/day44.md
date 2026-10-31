# Day 44: MIG

**MIG** (Multi-Instance GPU) ek badi GPU ko hardware slices mein kaat ta hai. A100 aur H100 pe ek slice ki apni memory aur compute hoti hai.

Do chhote models ek doosre ko poori GPU ka bandwidth nahi khaate. Slice sizes fixed profiles hain, jaise `1g.10gb`. Yeh software ka time-share nahi hai. Hardware partition hai.

**Rule:** MIG isolation deta hai. Slice ka size profile se aata hai, pod ke request se invent nahi hota.
