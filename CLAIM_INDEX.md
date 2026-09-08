# Claim-to-artifact index

This table gives the shortest audit path for each central result. All commands
are run from the repository root.

| Paper claim | Independent unit | Inputs | Reproduction | Primary output |
|---|---:|---|---|---|
| Natural Ouro path control grows from K1 to K3 on 2Wiki and MuSiQue | paired question | `data/analysis_ready/natural/ouro26_*.json` | `python analysis/analyze_scale_invariant_natural.py` | `scale_invariant_path_control.json` |
| Natural LoopUS path control grows from K1 to K8 on two independently constructed panels | paired question | `data/analysis_ready/natural/loopus_{2wiki,musique}.json` | same command | same output |
| Complete natural exit curves and coherent-repair fractions | paired question | all four endpoint natural files | `python analysis/analyze_natural_depth_curves.py` | `natural_depth_curves.json` |
| LoopUS natural response localizes early and then saturates | paired question | `data/analysis_ready/natural/loopus_full_depth_*.json` | `python analysis/analyze_loopus_full_depth.py` | `loopus_full_depth.json` |
| Ouro acquires exact-choice control over arbitrary fictional identities from K1 to K4 | fictional world | generated panel plus `fictional_ouro/` shards | `python tools/reproduce.py --only fictional-ouro` | `fictional_ouro/nonce_path_control_ouro26.json` |
| LoopUS-8B starts with strong choice control; later loops sharpen margin without detectable choice acquisition | fictional world | generated panel plus `fictional_loopus8/` shards | `python tools/reproduce.py --only fictional-loopus` | `fictional_loopus8.json` |
| The same edit has a large on-path effect and a small off-path effect | held-out fictional world | structural panel plus `structural/` shards | `python tools/reproduce.py --only structural` | `structural_falsifiers.json` |
| Graph connectivity remains influential when physical locality favors the other answer | held-out fictional world | same as above | same command | same output |
| Exit supervision, rather than tying alone, changes when control becomes readable in the controlled model | training seed, then held-out world | 32 factorial run files | `python tools/reproduce.py --only factorial` | `training_factorial.json` |

`python tools/build_claim_table.py` extracts the headline estimates and
intervals into a compact JSON and CSV crosswalk. `make reproduce` invokes it
automatically after all analyses complete.

Secondary output-boundary tests, natural-language corroboration, applicability
screens, and stopped branches are indexed in
`data/frozen_summaries/INDEX.json`. They are preserved as supporting evidence,
not silently promoted to confirmatory results.

