#!/bin/sh
# Contract test for #33: default-ruleset.json must match the spec that
# issue #33 asked for, encoded as assertions rather than left to prose so a
# future edit to the file can't silently drift from what a new repo is
# supposed to get. See CLAUDE.md §7 (wiring gets a contract test on the
# config's effect) and .github/repo-settings.md.
#
# Run directly: sh .github/default-ruleset.test.sh

set -eu

dir="$(cd "$(dirname "$0")" && pwd)"
ruleset_json="$dir/default-ruleset.json"

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

assert_eq() {
  desc="$1"
  want="$2"
  got="$3"
  if [ "$want" = "$got" ]; then
    assert "$desc" 0
  else
    printf '  want: %s\n  got:  %s\n' "$want" "$got" >&2
    assert "$desc" 1
  fi
}

jq -e . "$ruleset_json" >/dev/null
assert "default-ruleset.json is valid JSON" $?

assert_eq "targets the default branch" \
  '["~DEFAULT_BRANCH"]' \
  "$(jq -c '.conditions.ref_name.include' "$ruleset_json")"

assert_eq "enforcement is active" \
  "active" \
  "$(jq -r '.enforcement' "$ruleset_json")"

assert_eq "bypass list is exactly repo admin, PR-only" \
  '[{"actor_type":"RepositoryRole","actor_id":5,"bypass_mode":"pull_request"}]' \
  "$(jq -c '.bypass_actors' "$ruleset_json")"

assert_eq "restricts deletion" \
  "1" \
  "$(jq '[.rules[] | select(.type == "deletion")] | length' "$ruleset_json")"

assert_eq "blocks force pushes" \
  "1" \
  "$(jq '[.rules[] | select(.type == "non_fast_forward")] | length' "$ruleset_json")"

pr_rule="$(jq -c '.rules[] | select(.type == "pull_request") | .parameters' "$ruleset_json")"
assert "requires a pull request before merging" "$([ -n "$pr_rule" ] && echo 0 || echo 1)"
assert_eq "requires 0 approvals" \
  "0" \
  "$(printf '%s' "$pr_rule" | jq -r '.required_approving_review_count')"
assert_eq "requires extra approval for unattributed Copilot PRs" \
  "true" \
  "$(printf '%s' "$pr_rule" | jq -r '.require_extra_approval_for_unattributed_changes')"
assert_eq "squash is the only allowed merge method" \
  '["squash"]' \
  "$(printf '%s' "$pr_rule" | jq -c '.allowed_merge_methods')"

checks_rule="$(jq -c '.rules[] | select(.type == "required_status_checks") | .parameters' "$ruleset_json")"
assert "requires status checks to pass" "$([ -n "$checks_rule" ] && echo 0 || echo 1)"
assert_eq "requires branches to be up to date before merging" \
  "true" \
  "$(printf '%s' "$checks_rule" | jq -r '.strict_required_status_checks_policy')"
assert_eq "does not require status checks on creation" \
  "true" \
  "$(printf '%s' "$checks_rule" | jq -r '.do_not_enforce_on_create')"

assert_eq "requires CodeQL: errors, and high-or-higher security alerts" \
  '[{"tool":"CodeQL","alerts_threshold":"errors","security_alerts_threshold":"high_or_higher"}]' \
  "$(jq -c '.rules[] | select(.type == "code_scanning") | .parameters.code_scanning_tools' "$ruleset_json")"

exit "$fail"
