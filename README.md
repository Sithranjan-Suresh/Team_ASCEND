# ASCEND

Self-generated difficulty curricula for LLM code generation, extending
[LADDER](https://arxiv.org/abs/2503.00735) (Simonds & Yoshiyama, 2025) from
mathematical integration to Python programming puzzles ([P3](https://github.com/microsoft/PythonProgrammingPuzzles)).

## Status

This project began as a team submission for ACM Research at UT Dallas
(Spring 2025), presented as a poster. That phase is complete and its work
is preserved on the `adrian`, `eli-branch`, `henry`, `sith`, and
`integration-v1` branches — see [CREDITS.md](CREDITS.md) for who built what.

With approval from the research lead, development continues independently
on the `ascend-v2` branch. See `ASCEND: Forensic Audit & Version 2 Research
Plan` (project doc) for the full audit of the original implementation and
the research plan driving this continuation.

## Original idea

Apply LADDER's recursive variant-generation + GRPO reinforcement learning
methodology to code generation puzzles, using `assert sat(sol())` execution
as the correctness verifier in place of LADDER's numerical integration
check.

