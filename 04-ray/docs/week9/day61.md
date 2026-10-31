# Day 61: Value, reference, Object ID

Chhota number **pass by value** hai. Bytes task ke saath chali jaati hain.

Bada array **pass by reference** hai. Caller ko **ObjectRef** milta hai, jise Object ID bhi kehte hain. Worker Plasma se array padhta hai. Copy tab hoti hai jab object doosri node pe chahiye.

Plasma objects **immutable** hain. Update ka matlab naya object hai. Isliye kai readers ek writer ke race ke bina padh sakte hain. Purana ObjectRef purana array dikhata hai.

**Rule:** bada data ObjectRef hai. `ray.get` use value banata hai. Update naya ID hai, purani memory mein likhna nahi.
