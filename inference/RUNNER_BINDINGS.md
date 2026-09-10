# Runner bindings

The score artifacts retain the SHA-256 digest of the source that created them.
Most distributed runners are byte-identical to that source. Two need a
release-local adaptation because a private path was removed from the
structural panel and model adapters are not vendored.

| Runner | Artifact-bound source | Distributed source | Difference |
|---|---|---|---|
| Fictional Ouro scheduler | `19c9559c23ae4366420e925b8420ec576b90c26fe3df1018af2eab78ba650e47` | `eb736c78fcd33243db7832f5da68e3252ad83ec64e972e1b1e2ea04b8acec857` | Private repository discovery replaced by explicit `EVIDENCE_LOOPS_ADAPTER_DIR`; prompt schedule and scoring are unchanged |
| LoopUS-8B treatment | `cf8202a65f0ae670ecb05cae6f7044b3486ac5c94a2904d12ef673fb1ef981bc` | same | none |
| LoopUS-8B shared backend | `353ef265e101b5aa853bfcadd7cf428bddc7d83b03ca13e3c9e8fba801e68501` | same | none |
| HRM linear CPU analysis | `ef2ec09fceef49818195fd577fa4cdd0105a0bb70f5f552348df2e3d034cf084` | See `MANIFEST.sha256` for `analyze_hrm_linear.py` | Release analysis recomputes the gate and treatment from raw A/B logits without private runner paths |
| HRM branching CPU analysis | `23e4ebf79474a53f6e61768337b0674fe3ba8323358dc4e450a345d5b2a7fdfa` | See `MANIFEST.sha256` for `analyze_hrm_branching.py` | Release analysis retains the frozen interaction, gate, paired bootstrap, and decision sequence without private runner dependencies |
| Structural falsifiers | `db65b1c8a8be197000f1f88698245fcec2e9aac080acd142f00da3c75fd14fe9` | `79ae79c8b99777378c1e66afe64c2595abb02edb4e993397f421a83548335a34` | File and canonical constants rebound to the released panel after its private parent path was replaced; cells are unchanged |
| Surface-orbit Ouro scheduler | `85697fb85ba8a856ea93314212057940fe9cc3119eb4a440ad3e18d4ce696223` | See `MANIFEST.sha256` for `run_ouro_surface_orbit.py` | Release-local scheduler reads the exact public panel and reuses the distributed Ouro backend; private workspace discovery was removed |
| Surface-orbit CPU analysis | source runner chain bound in `data/PROVENANCE.json` | See `MANIFEST.sha256` for `analyze_surface_orbit.py` | Standalone release adaptation; estimands, 16-block whole-world bootstrap, seed, and decisions are unchanged |
| Answer-boundary GPU replay | `5931f1eef4892ba89f77c7006bc81fe0c8305ae38032db94e25ce3791d6625d3` | not distributed | Licensed natural prompts and decoded tokens are excluded; source result and runner hashes remain bound in provenance |
| Answer-boundary CPU analysis | `f7393dc902ca1cedd23478c27cfcfbe6be3fc58fae608e6925956f7efd00a408` | See `MANIFEST.sha256` for `analyze_decoding_boundary.py` | Text-free adaptation retains tie handling, conditional denominators, ranks, categories, and the original independent-unit bootstraps |
| Controlled factorial | `f65ee74fdc4c7bbbedd3c2acce71cc1f9dc9b48847f6f512149c007b2c714145` | same | none |

The initial LoopUS competence artifact records an earlier competence-only
runner digest. The treatment was executed with the later shared backend listed
above, and both competence and treatment rows are distributed. Statistical
reproduction reads the rows rather than trusting any runner digest.
