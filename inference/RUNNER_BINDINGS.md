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
| Structural falsifiers | `db65b1c8a8be197000f1f88698245fcec2e9aac080acd142f00da3c75fd14fe9` | `79ae79c8b99777378c1e66afe64c2595abb02edb4e993397f421a83548335a34` | File and canonical constants rebound to the released panel after its private parent path was replaced; cells are unchanged |
| Controlled factorial | `f65ee74fdc4c7bbbedd3c2acce71cc1f9dc9b48847f6f512149c007b2c714145` | same | none |

The initial LoopUS competence artifact records an earlier competence-only
runner digest. The treatment was executed with the later shared backend listed
above, and both competence and treatment rows are distributed. Statistical
reproduction reads the rows rather than trusting any runner digest.

