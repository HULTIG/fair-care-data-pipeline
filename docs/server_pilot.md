# Server pilot

Run the one-seed validation gate from the repository root:

```bash
python experiments/scripts/run_server_pilot.py \
  --datasets compas,adult,german,nij \
  --configs baseline,configa,configb,configc,default \
  --seed 42 \
  --output results/server-pilot
```

The command creates an immutable pilot directory containing a running manifest,
one directory per dataset/configuration, pipeline summaries or failure records,
and a final `manifest.json`. A pilot is complete only when every expected run
succeeds. Publication eligibility remains separate from run success and is
recorded for each result.
