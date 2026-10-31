# Day 42: NVIDIA Container Toolkit

Container ko GPU apne aap nahi dikhti. Driver host pe rehta hai. Container ke andar `nvidia-smi` tab fail hota hai jab GPU libraries mount na hon.

**NVIDIA Container Toolkit** woh libraries container mein mount karta hai. Driver copy nahi hota. Host ka driver rehta hai. Toolkit use container ko dikhata hai.

Bina toolkit ke pod `nvidia.com/gpu` maang bhi le, process GPU se baat nahi kar pati.

**Rule:** toolkit libraries deta hai. Driver host pe rehta hai.
