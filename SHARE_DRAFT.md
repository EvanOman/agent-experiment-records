# Unsent post draft

Asked Codex to beat 95% on MNIST without a training loop. It designed a stroke/template classifier, authored its parameters, and iterated through 26 candidates. Result: 96.03% on a fresh 10k holdout.

Code, sanitized transcript, CI replay:
https://github.com/EvanOman/agent-experiment-records

---

The linked experiment explains the important qualification: these are authored geometric templates and shared coefficients expanded into tensors, not an ordinary CNN with every weight matrix entry typed by the model. The holdout is a fresh split of MNIST's training partition, not the canonical benchmark test partition. The post is a draft for the repository owner to review and send.
