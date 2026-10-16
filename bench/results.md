| config | idle VRAM | peak VRAM | RTF | first chunk | worst seam | line lengths, s |
|---|---|---|---|---|---|---|
| stock | 3.16 GB | 3.59 GB | 0.66 | 604 ms | 1.19 | 4.7 / 4.4 / 9.2 / 5.1 / 5.8 |
| offload-repack | 1.83 GB | 3.25 GB | 0.67 | 627 ms | 1.19 | 4.7 / 4.4 / 9.2 / 5.1 / 5.8 |
| offload-repack-ctx25 | 1.83 GB | 2.01 GB | 0.53 | 425 ms | 1.12 | 4.7 / 4.4 / 9.2 / 5.1 / 5.8 |
| optimized | 1.84 GB | 2.02 GB | 0.52 | 428 ms | 1.12 | 4.7 / 4.4 / 9.2 / 5.1 / 5.8 |
| optimized-int8 | 1.51 GB | 1.67 GB | 0.79 | 639 ms | 1.12 | 4.3 / 4.7 / 8.9 / 4.1 / 6.2 |
| optimized-int4 | 1.23 GB | 1.53 GB | 0.36 | 315 ms | 2.32 | 3.8 / 5.5 / 9.8 / 22.6 / 6.6 |

GPU: NVIDIA GeForce RTX 4060. VRAM = whole-GPU memory.used delta from before the model loaded. RTF = generation time / audio length. Seam > ~3 would mean a click at a chunk join.
