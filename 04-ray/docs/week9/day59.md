# Day 59: GCS, raylet, Plasma

| Piece | Kaam | Socho |
|---|---|---|
| **GCS** (Global Control Store) | Cluster ki directory. Nodes, actors, object locations. | Phone book |
| **Raylet** | Har node pe agent. Local resources. | Building ka guard |
| **Local scheduler** | Is node pe kaun sa task ab chalega. | Guard ki list |
| **Object store (Plasma)** | Bada shared-memory store. Arrays yahan, process ke heap mein nahi. | Shared godown |

Task ka code chhota ho sakta hai. Array bada hota hai. Array Plasma mein rehta hai, taaki do processes use copy kiye bina padh saken, jab tak woh same node pe hon.

**Rule:** GCS naam rakhta hai. Raylet node chalaata hai. Plasma bytes rakhta hai.
