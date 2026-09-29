# Credits

ASCEND was originally built as a team project for ACM Research at UT Dallas
(Spring 2025), applying the [LADDER paper](https://arxiv.org/abs/2503.00735)
(Simonds & Yoshiyama, Tufa Labs) to automated code-generation problem
curricula. The original team and their contributions (see the `adrian`,
`eli-branch`, `henry`, and `sith` branches for full history):

- **Henry Harrison** — GRPO reinforcement learning implementation
  (`grpo.py`, `ladder_optimizer.py` and its iterations), training pipeline
  architecture, and the original variant-generator prototype.
- **Eli** (`Fa2888` / `eli-branch`) — AST-based difficulty estimator
  (`diff_estimator.py`), including the relative-scoring refinement.
- **Adrian Hautea** — variant usefulness classifiers (MLP and logistic
  regression), feature extraction experiments.
- **Sithranjan Suresh** (`sith`) — P3-based and LeetCode-style variant
  generation pipelines (`ascend_generate.py`, `variants.py`).
- **Ilayamudhan Elango Umamaheshwari, Abhijith Ulta** — team contributions
  per the [original poster](https://github.com/Abhijith-utla/Team_ASCEND).

The team's work concluded with a poster presentation (Spring 2025). With
approval from the research lead, this fork continues ASCEND as independent
research — see `README.md` for the current scope and `ascend-v2` branch for
ongoing work. All original branches and commit history are preserved
unmodified in this fork for attribution.
