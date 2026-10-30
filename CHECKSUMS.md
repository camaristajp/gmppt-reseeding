# Checksums

MD5 of every released file. The twelve under *Pinned* are the configuration
that produced the reported results; they match the freeze record.

## Pinned

| MD5 | Released path | Original path |
|---|---|---|
| `e01eb232b3c7b0550dcd95f53974f4fb` | `phase2/p12_relocation_comparison.py` | `phase2/p12_relocation_comparison.py` |
| `6e63114e98cd0005009392f20b5719e5` | `phase2/p13_gradual_sweep.py` | `phase2/p13_gradual_sweep.py` |
| `f39c75972deebf75912fabdc542e5a03` | `phase2/p14_composite_spliced.py` | `phase2/p14_composite_spliced.py` |
| `682002d6254da16fe59ce6a34487a6d0` | `phase2/transition.py` | `phase2/transition.py` |
| `42629c6989ce9cd3a56eeadc86610964` | `gmppt/hybrid.py` | `gmppt/hybrid.py` |
| `8617b8203f3745bb444b0ab293e41bc4` | `gmppt/datasheet.py` | `gmppt/datasheet.py` |
| `9cd7b7df4d098ac696b878a4b46fec4a` | `gmppt/harness.py` | `gmppt/harness.py` |
| `2f714e09bf7c32d1064fbf5d14f304b4` | `gmppt/model.py` | `gmppt/model.py` |
| `6e106cce2341e458186d370c98c60bf7` | `models/c3_two_stage_design.pkl` | `results/phase3/c3_two_stage_design.pkl` |
| `fdaeda672a50e2f7574179bc2b50c36c` | `models/jamaa_net_keepbest.joblib` | `results/phase3/jamaa_net_keepbest.joblib` |
| `15d8f8fde682a67e87ca57b3eed3198e` | `splits/design_split.json` | `paper/splits/design_split.json` |
| `b4c1db83503e225861e32d4f8c106d6f` | `splits/sealed_test_split.json` | `paper/splits/sealed_test_split.json` |
| `3130130b3a851ac209e5a3fb6ee1513a` | `results/R12a_sealed_p12.json` | `runs_archive/R12a_sealed_p12.json  (scrubbed)` |
| `5bd1aea9fc12802af352a5c3f5e8872c` | `results/R12b_sealed_p14.json` | `runs_archive/R12b_sealed_p14.json  (scrubbed)` |
| `9a03fe958a920556072ad526a64544a2` | `results/R12c_sealed_p13.json` | `runs_archive/R12c_sealed_p13.json  (scrubbed)` |

## Divergence from the 28 October freeze record

The freeze record lists a different MD5 for the files below. Those
hashes match no file in this repository or its git history and were
recorded on a second machine. The hashes above are of the files
actually released. The run configuration itself is recorded inside
each archived result file and matches the freeze record in full.

| File | Freeze record | Released |
|---|---|---|
| `gmppt/hybrid.py` | `3d679c9933b99d7142c60ea54f9d9ff5` | `42629c6989ce9cd3a56eeadc86610964` |
| `gmppt/datasheet.py` | `0b87fcefc09463c4eadf4c673ad48865` | `8617b8203f3745bb444b0ab293e41bc4` |
| `gmppt/model.py` | `841ada4685002183353cc82272a6a957` | `2f714e09bf7c32d1064fbf5d14f304b4` |

Verify with:

```bash
md5sum -c <(awk -F'|' '/^\| `/ {gsub(/`| /,"",$2); gsub(/`| /,"",$3); print $2"  "$3}' CHECKSUMS.md)
```
