# Contributing

This repo follows the operating model in [`CLAUDE.md`](./CLAUDE.md). It applies to humans and agents equally — the rules are about the work, not the worker.

## Before you start

1. Read [`CLAUDE.md`](./CLAUDE.md) end-to-end. It is the contract.
2. Read the relevant spec in [`docs/specs/`](./docs/specs/). No spec → write one first.
3. Skim [`docs/decisions/`](./docs/decisions/) for ADRs that constrain your area.

## The loop

**Spec → Plan → Tests → Code → Ship.** No skipping for "small" work above S. Work is sized Xs/S/M/L/Xl — see [`docs/sizing.md`](./docs/sizing.md) and `CLAUDE.md` §3–§4.

## Branching & PRs

- Branch from `main`. Squash-merge to `main`. Branches never touch other branches.
- Branch name: `{type}/{scope}-{description}`, kebab-case.
- PR title: Conventional Commits (`{type}({scope}): {imperative}`).
- Fill every section of the PR template. No placeholders.
- No AI attribution anywhere in git history.

## Dependencies

`requirements.txt` / `requirements-dev.txt` are hash-pinned locks compiled from
`requirements.in` / `requirements-dev.in` (ADR-0013) — don't hand-edit the `.txt` files.
CI installs with `pip install --require-hashes`, which refuses to install anything not in
the lock.

To bump a version or accept a Dependabot PR, edit the `.in` file, then relock:

```
uv pip compile requirements.in --generate-hashes --python-version 3.14 \
    --python-platform x86_64-manylinux_2_28 -o requirements.txt
uv pip compile requirements-dev.in --generate-hashes --python-version 3.14 \
    --python-platform x86_64-manylinux_2_28 -o requirements-dev.txt
```

New/major dependency bumps still need the approval `CLAUDE.md` §6 describes; relocking
doesn't skip that.

## Tests

Tests are the contract. See `CLAUDE.md` §7. A PR without appropriate tests is not done.

## Issues

Out-of-scope bugs and tech debt go in issues, not in your current PR. Use the templates in `.github/ISSUE_TEMPLATE/`. Title every issue `[Px][Size] title` and record its size with a one-line justification — see [`docs/sizing.md`](./docs/sizing.md) for the criteria.

## Landmines

When you (or an agent) hit a recurring miss, append to `CLAUDE.md` §10. Daily review for the first 60 days.

## Code of conduct

Be direct. Be kind. Don't ship what you wouldn't review.
