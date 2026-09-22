#!/bin/sh
# Contract test for #31/#71/#72: merge-train.yml must not poll on a schedule,
# and once queue:ready is adopted as a real event trigger it must actually be
# wired to advance the queue — not left dispatch-only while PRs sit labelled.
# See merge-train.yml's header for the full rationale.
#
# Run directly: sh .github/workflows/merge-train.test.sh

set -eu

workflow_dir="$(cd "$(dirname "$0")" && pwd)"
train="$workflow_dir/merge-train.yml"

fail=0

assert() {
  desc="$1"
  cond="$2"
  if [ "$cond" = "0" ]; then
    printf 'ok: %s\n' "$desc"
  else
    printf 'FAIL: %s\n' "$desc"
    fail=1
  fi
}

# 1. No schedule: trigger — polling stays off regardless of how the queue is
# otherwise wired; the event triggers below are the intended replacement.
if grep -qE '^\s*schedule:' "$train"; then
  assert 'merge-train.yml declares no schedule: trigger' 1
else
  assert 'merge-train.yml declares no schedule: trigger' 0
fi

# 2. workflow_dispatch stays — the mechanism must remain manually testable.
if grep -qE '^\s*workflow_dispatch:' "$train"; then
  assert 'merge-train.yml keeps workflow_dispatch' 0
else
  assert 'merge-train.yml keeps workflow_dispatch' 1
fi

# 3. queue:ready is wired to an event trigger in merge-train.yml itself: a
# PR gaining the label, or a push to main, must advance the queue without
# waiting for someone to run workflow_dispatch by hand.
if grep -qE '^\s*types:\s*\[.*labeled.*\]' "$train"; then
  assert 'merge-train.yml triggers on pull_request labeled' 0
else
  assert 'merge-train.yml triggers on pull_request labeled' 1
fi

if grep -qE '^\s*push:' "$train" && grep -qE '^\s*branches:\s*\[.*main.*\]' "$train"; then
  assert 'merge-train.yml triggers on push to main' 0
else
  assert 'merge-train.yml triggers on push to main' 1
fi

exit "$fail"
