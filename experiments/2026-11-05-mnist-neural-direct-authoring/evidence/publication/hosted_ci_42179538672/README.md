# Hosted replay evidence

GitHub Actions reproduced **v087 at 9,461/10,000** using the public canonical test archive and committed source `6090bbbc9c2588ab5ebbc3e15197970006ffc530`. The saved aggregate binds the checkpoint, inference source and checksum-verified archive. The original 95% holdout goal remains unmet. [Hosted run](https://github.com/EvanOman/agent-experiment-records/actions/runs/42179538672)

The offline checks and v095 nonglyph smoke passed. The labels-only split check matched both original development index hashes. Full public v095 validation replay remains pending.

The four result JSON files preserve the exact downloaded artifact bytes. `artifact_metadata.json` records GitHub's archive digests, artifact identifiers and expiry dates. These checked-in JSON files retain the aggregate evidence beyond hosted artifact retention. The original two small ZIP archives are also preserved. `receipt_at_archival.json`, `job_metadata_at_archival.json` and the native review describe the verification scope. At archival time both experiment jobs had passed and the overall workflow was still running. The later completion record confirms success for the entire workflow. [Completion record](workflow_completion.json)

The initial workflow failed YAML parsing before any job started. Its failure and the reviewed correction remain in the adjacent publication evidence. [Initial failed run](https://github.com/EvanOman/agent-experiment-records/actions/runs/42178932814)
