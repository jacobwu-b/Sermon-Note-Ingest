#!/bin/sh
# Contract test for the zero-major-PR fix: dependabot-reconciliation.yml must
# not create a tracking issue when no `dependencies:major` PRs are open, and
# must update (not leave stale) an existing month's issue when the count
# drops to zero after it was opened. See CLAUDE.md §7 (wiring gets a
# contract test on the config's effect).
#
# Run directly: sh .github/workflows/dependabot-reconciliation.test.sh

set -eu

workflow_dir="$(cd "$(dirname "$0")" && pwd)"
workflow="$workflow_dir/dependabot-reconciliation.yml"

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

# Extract the run: block's shell script and execute it under a mocked `gh`
# and `jq` so the workflow's actual logic runs, not a re-description of it.
script_dir="$(mktemp -d)"
trap 'rm -rf "$script_dir"' EXIT

extract_run_script() {
  awk '
    /^        run: \|/ { capture=1; next }
    capture && /^      - name:/ { exit }
    capture { print }
  ' "$workflow" | sed 's/^          //'
}
extract_run_script > "$script_dir/run.sh"

# Mock `gh`, driven by two env vars the test sets per scenario:
#   MOCK_PR_COUNT     — number of open major PRs `gh pr list` reports
#   MOCK_EXISTING_NUM — issue number `gh issue list` reports as already open
#                        for this month (empty = none)
# Every invocation is logged so assertions can check what ran.
cat > "$script_dir/gh" <<'MOCK'
#!/bin/sh
log="$MOCK_LOG"
echo "gh $*" >> "$log"
case "$1" in
  pr)
    n="${MOCK_PR_COUNT:-0}"
    if [ "$n" -eq 0 ]; then
      echo "[]"
    else
      python3 -c "import json,sys; print(json.dumps([{'number': i, 'title': 't', 'url': 'u'} for i in range(1, int(sys.argv[1]) + 1)]))" "$n"
    fi
    ;;
  issue)
    case "$2" in
      list) echo "${MOCK_EXISTING_NUM:-}" ;;
      edit|create) : ;;
    esac
    ;;
esac
MOCK
chmod +x "$script_dir/gh"

run_scenario() {
  pr_count="$1"
  existing_num="$2"
  log="$script_dir/log-$3"
  : > "$log"
  # As Actions runs a `run:` step: bash -e. The step is bash (pipefail, <<<),
  # and `sh` is dash on ubuntu-latest, where it fails before doing anything.
  MOCK_PR_COUNT="$pr_count" MOCK_EXISTING_NUM="$existing_num" MOCK_LOG="$log" \
    GH_TOKEN=fake REPO=owner/repo \
    PATH="$script_dir:$PATH" \
    bash -e "$script_dir/run.sh" > "$script_dir/stdout-$3" 2>&1 || true
}

# 1. No open majors, no existing issue this month → no issue is created.
run_scenario 0 "" "no-majors-no-existing"
if grep -q '^gh issue create' "$script_dir/log-no-majors-no-existing"; then
  assert 'zero majors + no existing issue: no issue created' 1
else
  assert 'zero majors + no existing issue: no issue created' 0
fi

# 2. No open majors, an issue already exists for this month → it's edited
# to say nothing is outstanding, not left with a stale checklist, and no
# new issue is created.
run_scenario 0 "42" "no-majors-existing"
if grep -q '^gh issue create' "$script_dir/log-no-majors-existing"; then
  assert 'zero majors + existing issue: no new issue created' 1
else
  assert 'zero majors + existing issue: no new issue created' 0
fi
if grep -q '^gh issue edit 42' "$script_dir/log-no-majors-existing"; then
  assert 'zero majors + existing issue: existing issue is updated' 0
else
  assert 'zero majors + existing issue: existing issue is updated' 1
fi

# 3. Majors open, no existing issue → unchanged: a new issue is created.
run_scenario 2 "" "majors-no-existing"
if grep -q '^gh issue create' "$script_dir/log-majors-no-existing"; then
  assert 'nonzero majors + no existing issue: issue created (unchanged)' 0
else
  assert 'nonzero majors + no existing issue: issue created (unchanged)' 1
fi

# 4. Majors open, existing issue → unchanged: it's updated with the checklist.
run_scenario 3 "42" "majors-existing"
if grep -q '^gh issue edit 42' "$script_dir/log-majors-existing"; then
  assert 'nonzero majors + existing issue: issue updated (unchanged)' 0
else
  assert 'nonzero majors + existing issue: issue updated (unchanged)' 1
fi
if grep -q '^gh issue create' "$script_dir/log-majors-existing"; then
  assert 'nonzero majors + existing issue: no duplicate issue created' 1
else
  assert 'nonzero majors + existing issue: no duplicate issue created' 0
fi

exit "$fail"
