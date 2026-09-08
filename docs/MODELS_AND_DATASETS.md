# Models and datasets

## Recurrent systems

| System | Repository | Frozen revision | Exits used | Precision |
|---|---|---|---|---|
| Ouro-2.6B | `ByteDance/Ouro-2.6B` | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | K1, K2, K3, K4 | bfloat16 |
| LoopUS Qwen3-1.7B SFT | upstream LoopUS release | `469f03291c3b` | K1 through K8 | bfloat16 |
| LoopUS Qwen3-8B | `Thrillcrazyer/Qwen3-8B_LoopUS` | `4424cfd8e36c77fee337a04602e3ddf6faae533f` | K1, K2, K4, K8 | bfloat16 |

K is compared only within a system. Ouro exposes native intermediate exits
from one weight-tied recurrent trajectory. LoopUS performs a separate
fixed-depth forward with adaptive stopping disabled. Runtime checks verify the
requested number of reasoning-block applications. Equal K values across the
two families are not treated as equal compute.

The controlled graph-memory factorial uses four applications in every cell.
Tied cells reuse one block; untied cells use four initially identical blocks.
Supervision is attached either only at K4 or at every exit. The comparison is
operation-matched, not parameter-matched.

## Natural data sources

The natural panels were constructed from 2WikiMultiHopQA and MuSiQue. Their
original terms govern access and redistribution. The release therefore
contains no benchmark questions, answer strings, evidence prose, or source
item identifiers. It contains the paired model scores and grouping variables
needed for statistical reproduction.

## Generated data

The fictional panel contains 288 arbitrary worlds, six relation families, 12
ordered terminal pairs per family, and four disjoint two-hop paths per world.
Graph state, query arm, and candidate order are fully crossed. An exact
symbolic solver verifies the licensed answer, bridge-swap reversal, coherent
repair, and distractor disconnection for every cell.

The structural panel adds answer-path versus disconnected edits and matched
versus crossed physical locality on a held-out 144-world split.

