# Models and datasets

## Recurrent systems

| System | Repository | Frozen revision | Exits used | Precision |
|---|---|---|---|---|
| Ouro-2.6B | `ByteDance/Ouro-2.6B` | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | K1, K2, K3, K4 | bfloat16 |
| LoopUS Qwen3-1.7B SFT | upstream LoopUS release | `469f03291c3b` | K1 through K8 | bfloat16 |
| LoopUS Qwen3-8B | `Thrillcrazyer/Qwen3-8B_LoopUS` | `4424cfd8e36c77fee337a04602e3ddf6faae533f` | K1, K2, K4, K8 | bfloat16 |
| HRM-Text-1B | `sapientinc/HRM-Text-1B` | `1f82ac2b71222f0c100a224a33f24b44a3000b6d` | retained H1 and native H2 | bfloat16 |

K is compared only within a system. Ouro exposes native intermediate exits
from one weight-tied recurrent trajectory. LoopUS performs a separate
fixed-depth forward with adaptive stopping disabled. Runtime checks verify the
requested number of reasoning-block applications. Equal K values across the
two families are not treated as equal compute.

HRM-Text couples a slow H module to a fast L module. The released checkpoint
uses two H cycles and three L cycles within each H cycle. H2 is its native
final state. H1 is retained from the same stock H2 trajectory and scored with
the released language-model head after full-vocabulary H2 parity is verified;
it is not a trained early exit. H and K are not equated across model families.

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

The HRM branching panel contains 48 gate worlds and 144 different confirmation
worlds. Each bridge has two outgoing relation types, and the requested relation
selects the relevant terminal. The generated panel fully crosses source arm,
requested relation, graph state, answer order, and H readout.
