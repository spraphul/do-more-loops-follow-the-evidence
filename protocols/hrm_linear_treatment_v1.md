# HRM-Text-1B fictional path-control treatment v1

Status: frozen after the original-only competence gate passed and before any
HRM bridge-swap or topology-repair output was accessed.

The treatment is unlocked only by competence artifact SHA-256
`fecebbb0d0b2302d61270b20b5d6ecb13a898a187757ef06deac94d98b1d52e3`.
That artifact reports H2 intact accuracy 0.908854 with a stratified-world 95%
interval [0.867188, 0.947917], H1 descriptive accuracy 0.500000, exact stock-H2
full-vocabulary parity, and `PROMOTE_TO_TREATMENT`.

## Question and fixed outcomes

Does the graph-directed response emerge from H1 to H2 in HRM-Text-1B, a nested
dual-timescale recurrent architecture distinct from Ouro and LoopUS?

The primary experiment uses the 240 nonce worlds whose HRM outcomes were not
accessed by the competence gate, exactly 40 from each relation family. The 48
competence worlds are excluded from treatment and primary inference. Their
sorted complement has SHA-256
`480dfe02403f544654e73b89524aefd7881e624bd5dca94cb0f7b3a6d505b123`.
Original and bridge-swap cells are read at retained H1 and stock H2. Coherent
repair is read at H2. The prompt adapter, PrefixLM mask, checkpoint, panel, and
A/B scoring boundary are identical to the competence run. One unchanged
stock-H2 forward supplies both H states; every shard must reproduce the
full-vocabulary H2 parity check before scoring H1.

The primary endpoint is the H2-minus-H1 exact-choice bridge-swap gain. The
raw-margin gain, H2 path contrast, H2 repair increment, and topology-recovery
fraction are fixed supporting estimands. The conclusion is tiered. A
depth-dependent bridge-swap response requires the 95% lower bound to exceed
zero for all three of: H2 path-margin contrast, H2-minus-H1 raw-margin gain,
and H2-minus-H1 exact-choice gain. The stronger topology-sensitive path-control
conclusion additionally requires the H2 repair-increment lower bound to exceed
zero. Resampling uses 10,000 whole-world draws stratified within the six
relation families, with 40 untouched worlds per family.

The stronger success tier establishes depth-dependent, topology-sensitive path
control in a third native recurrent family. The weaker tier establishes only
depth-dependent bridge-swap sensitivity. Neither makes H1 a trained exit,
identifies a neural circuit, or shows that all recurrent language models share
the same trajectory.

## Execution

Run eight 30-world shards under the same isolated Transformers 5.9 runtime
used for competence, then merge with
`analyze_hrm_text_nonce_treatment.py`. Each treatment shard must receive the
exact promoted competence artifact. No threshold or prompt change is permitted
after treatment begins.

Frozen bindings before treatment inference:

- treatment runner SHA-256:
  `4b55c4fe7871398e7987e94bfd9db8f4d6a0d312c30ac1d7abd7a66b42b75b80`
- competence adapter source SHA-256:
  `f1ee430807fa3129108c591505d40ead75b92da7137fba387d223aeaeb332970`
- checkpoint revision: `1f82ac2b71222f0c100a224a33f24b44a3000b6d`
- checkpoint weights SHA-256:
  `f8fe2b2bf6948414e8e8d6538659198726d98f967c55b533b7aabe8a1fa9a584`
- prompt-adapter SHA-256:
  `cfa4a8a238f1697a551bc073e9256f4619b0cf5c04cae7b6a70f4b0fd7cf6bda`

Each shard must record successful full-vocabulary H2 parity, the exact model
and adapter bindings above, and 600 scored rows. The primary merged grid is
4,800 rows over the untouched 240 worlds.
