# Validated pilot results

Before aggregation, consolidate the pilot manifests:

```bash
.venv/bin/python experiments/scripts/validate_pilot_manifests.py \
  --manifests results/server-pilot/pilot-20260910T155210Z-seed42/manifest.json,results/server-pilot-nij-2/pilot-20260910T155813Z-seed42/manifest.json \
  --output results/validated/server-pilot-manifest.json
```

The validator requires one successful, evidence-valid summary for every
dataset/configuration pair. It rejects missing runs, duplicate successful runs,
summary/run ID mismatches, and summaries with invalid evidence. The resulting
manifest is the only permitted input to publication aggregation.
