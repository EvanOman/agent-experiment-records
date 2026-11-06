# Failures and incomplete work retained in this snapshot

All completed scores, including declines and ties, appear in [scores.json](scores.json), [scores.csv](scores.csv) and the [complete candidate log](CANDIDATE_LOG.md). The structured history has 108 development results for 92 scored candidates. The 96-row candidate inventory also preserves unscored or reserved entries.

- **v003 declined:** its changed connectivity increased false cavities. The result was retained and the next revision branched from v002.
- **v015 verifier discrepancy:** a synthetic reference associated floating-point additions differently; the reference was corrected to the declared graph order without changing the candidate.
- **v022 construction failure:** an assertion caught a remembered coefficient inconsistent with the saved source row before freezing. The failed source was preserved locally; downstream evaluation refused the absent manifest before reading data. The later completed v022 score belongs to the corrected frozen artifact.
- **v027 failed synthetic verification:** diagnostic feature stacks had incompatible dimensions, so no development score was produced. v028 repaired the omitted diagnostic slots. v046 and v066 never produced scored classifiers; their rejected/runtime-only investigations remain explicitly marked.
- **Development interruptions:** v063 and v083 had interrupted first evaluations; v088, v089 and v090 also have interrupted execution histories. Unchanged development evaluations or missing durable batches were later completed. Causes of external termination are unknown. Recorded elapsed times for resumable jobs can include interruption gaps. No interrupted attempt was relabeled as a score, and this recovery policy was never used for sealed confirmation.
- **Canonical confirmation failed:** the once-only frozen v087 evaluation returned 9,461/10,000 (94.61%), below 9,500. Its clean review establishes no passing score. The consumed canonical set was never reused for adaptive development. Public inference replay is explicitly separate from that historical event.
- **Pending at cutoff:** v095 work evaluation was running and v096 had no completed score. Neither is assigned an invented result. No newer model or unused reserve content is in this snapshot.

This is a public summary of the retained source record, not the complete local execution archive. Source hashes, aggregate evidence and redaction/omission records distinguish included records from omitted runtime artifacts.
