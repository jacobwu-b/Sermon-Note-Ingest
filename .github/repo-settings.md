# GitHub Repository Settings

These settings are part of the contract. They enforce in GitHub what
`CLAUDE.md` enforces in process. Apply them to every new repo from this template.

GitHub does not load most of these from a file in the repo. Apply them via the
UI, the `gh` CLI, the REST API, or — ideally — the
[Probot Settings app](https://github.com/apps/settings) using
`.github/settings.yml` (provided alongside this doc).

If you change a setting, update this file in the same PR.

---

## General

| Setting | Value | Why |
|---|---|---|
| Default branch | `main` | Single trunk. |
| Wikis | Off | `docs/` is the wiki. |
| Issues | On | We file out-of-scope work as issues. |
| Projects | On | Optional, but useful. |
| Discussions | Off until needed | Avoid scattering decisions. |
| Allow forking | Off (private repos) | Reduces leak surface. |
| Sponsorships | Off | N/A for product repos. |

## Pull requests

| Setting | Value | Why |
|---|---|---|
| Allow merge commits | **Off** | `CLAUDE.md` §5 — squash only. |
| Allow squash merging | **On** | The only merge style. |
| Allow rebase merging | **Off** | Rewrites history; breaks the "born from main" invariant. |
| Default commit title for squash | "Pull request title" | PR title is the squash commit. |
| Default commit message for squash | "Pull request title and description" | Keeps context in history. |
| Always suggest updating PR branches | **On** | Surfaces drift before merge. |
| Allow auto-merge | **On** | Required for Dependabot auto-merge and for agents to auto-merge once checks pass + reviews approve. |
| Automatically delete head branches | **On** | No stale branches. |
| Merge queue (on `main`) | **Only if available** — see below | Serializes + rebases queued PRs against fresh `main`. Needs an org-owned public repo or Enterprise Cloud; confirm before documenting it as on. |

**No merge queue available (most repos on this template).** `.github/workflows/merge-train.yml`
is the substitute: label a ready, approved, green PR `queue:ready` and the workflow serializes it
onto `main` by hand, one at a time, roughly what a real queue would do. It requires the `queue:ready`
label (`settings.yml`) and `allow_auto_merge: true` (above). `queue-stall-alert.yml`
catches the failure mode the train can't predict — a PR that stops progressing for a reason the
train never evicts it for. Both are optional: skip them entirely if a real merge queue is available,
or if this repo's PR volume never queues more than one PR at a time.

## Branch protection — `main`

`main` is governed by a **Ruleset** (Settings → Rules → Rulesets), not the legacy
branch-protection block in `.github/settings.yml`. The switch is required because
a per-actor **review bypass for Dependabot** cannot be expressed in legacy
protection.

**Every value below is a starting point to be calibrated, not a target to reach
for.** A rule nothing can satisfy blocks everything, and the fastest way to make
things merge again is to disable enforcement wholesale — so the strictest-looking
ruleset is often the one that ends up switched off entirely. Set each rule to what
this repo's actual people and plan can meet today, and raise it in the same change
that adds the capacity to meet it.

The ruleset must enforce, **for everyone except the documented bypass**:

- **Require a pull request before merging**
  - Require approvals: **0 while the repo has one maintainer; 1 once a second
    person can review; 2 past five.** GitHub does not let anyone approve their
    own pull request, so on a solo repo any non-zero value blocks *every* PR
    rather than gating it. That is not a stricter setting, it is an
    unsatisfiable one. Where approvals are 0, name the human signal that stands
    in for review — a label applied by hand, a checklist — and be honest in this
    document that it is a convention, not a control.
  - Dismiss stale approvals on new commits: **On, once approvals > 0.** At 0
    there are no approvals to dismiss, so it enforces nothing while reading to
    the next person as a gate that is holding. Turn it on in the same change
    that raises the count.
  - Require approval of the most recent reviewable push: same — **On once
    approvals > 0**, a no-op before that.
  - Require review from Code Owners: **Off unless `CODEOWNERS` resolves to real,
    distinct reviewers.** A `CODEOWNERS` full of unexpanded placeholders is
    malformed, so GitHub assigns no owner to any path and the rule enforces
    nothing while this document claims it is on. On a solo repo every path's
    owner is the author, which routes review of his work back to him: a rubber
    stamp, not a gate. Ship the file inert (fully commented out) and turn both
    on together. Assert the two agree **in both directions**, so the rule cannot
    come back without the file, or the file without the rule.
  - **Bypass list: exactly the entries listed in this document, and nothing
    else.** An entry found live that is not listed here is drift, and an entry
    listed here that is missing live is an outage waiting for the next run that
    needs it. Every bypass is load-bearing or it is deleted; there is no
    third category. Scope a review bypass to **review only** — a bypass that
    also clears status checks lets its actor merge red.
    - **Dependabot**, so minor/patch PRs merge unattended. Every human PR still
      needs review.
    - **Any deploy key or App that automation pushes with.** These are the ones
      people forget, because nothing fails at the moment they are forgotten:
      the next PR still merges, and only the next unattended run discovers it,
      after it has already done the expensive part of its work. If you rebuild
      the ruleset, re-add them in the same sitting and then exercise the
      automation to confirm the push still lands.
      - Live: `DeployKey` / `always`, matching the write-access deploy key
        titled `poll-workflow-push` (repo Settings → Deploy keys), whose
        private half is the `DEPLOY_KEY` secret `.github/workflows/poll.yml`
        (and `.github/workflows/transcribe.yml`, for its own ledger commit)
        uses to push `data/` updates straight to `main`.
      - **Not a bypass-list concern:** `CONTENT_REPO_TOKEN`, used by
        `transcribe.yml` to push transcripts to the private Sermon-Note-Content
        repo, authenticates against *that* repo, not this one — it needs no
        entry here. It does need `contents:write` scope on Sermon-Note-Content,
        confirmed before the first real transcription run.
  - **A credential used by automation is two settings, and they drift apart
    silently:** the bypass or permission entry, and the secret itself. A key on
    the bypass list that no workflow holds cannot sign a push; a key the
    workflow holds that is not on the bypass list is refused. Create them in one
    sitting and check both. Where the automation can read the property — that a
    remote is what it should be, that a secret is non-empty — make it refuse
    early rather than discovering it after the spend.
- **Require status checks to pass**
  - Require branches to be up to date before merging: **On.** This is the
    plan-independent way to get the property a merge queue is usually wanted for
    — a PR tested against the base it *lands* on, not the one it branched from.
    It is serial and noisier under bursts of Dependabot PRs, and it is available
    everywhere.
  - Required checks: exactly the contexts something in this repo **actually
    reports**. Keep the list in one place that both the ruleset and a test can
    read, rather than restating it in several documents — restated lists are
    what drift. A job that is disabled (`if: false`) reports no check run, so
    requiring its name is a gate nothing can ever satisfy; add it in the same
    change that enables the job. The same applies to any rule that waits on a
    reporter: a code-scanning rule with no scanning workflow leaves every PR
    `blocked` while all checks show green.
  - **Changing this list is a rollout, not an edit.** The ruleset requires
    contexts by name and usually cannot be changed from CI. *Renaming*: narrow
    the live ruleset to the contexts that are not changing, merge, then widen it
    to the final list — either order done in one step deadlocks. *Removing* is
    sharper: narrow the live ruleset first, then merge, because a PR that has
    stopped reporting a still-required context can never satisfy it.
- **Require merge queue**: **only if the platform offers it to this repo.**
  Merge queue requires a public repository owned by an organization, or a
  private repository on Enterprise Cloud — so a user-owned private repo cannot
  have one at any price, and making it public does not help. Confirm
  availability before documenting it as on, and add `merge_group:` to the CI
  workflow in the same change that enables it, never before. An absent queue and
  an unpurchasable one look identical in the run history: zero queued runs reads
  the same for "misconfigured" and "not available here".
- **Require conversation resolution before merging**: **On**
- **Require signed commits**: **On** (raise this bar early; it's painful to add later)
- **Require linear history**: **On** (squash-only enforces this; belt + suspenders)
- **Block force pushes** and **Restrict deletions**: **On**
- **Restrict who can push to matching branches**: nobody
- Enforcement: **Active**, applies to admins (no org/admin bypass beyond those
  listed above).

> Prefer approvals **0** with the rest of the ruleset **on** over approvals 1
> with the ruleset off. The visible symptom of an unsatisfiable rule is that
> nothing merges, and the quickest way to make things merge again is to disable
> enforcement wholesale — which is how a repo ends up with no protection at all
> while its settings document still describes a strict one.

### Verifying — this document is not evidence

Nothing in this repo can *apply* the ruleset, so nothing in this repo should be
trusted to describe it. Before relying on a gate, query the live configuration,
and where automation depends on a gate, have that automation assert it and refuse
rather than proceed. Replace `<owner>/<repo>` with the repo slug.

```sh
# What main actually enforces.
gh api repos/<owner>/<repo>/rules/branches/main \
  --jq '[.[] | select(.type == "required_status_checks")
             | .parameters.required_status_checks[].context]'

# The staleness gate: must print true.
gh api repos/<owner>/<repo>/rules/branches/main \
  --jq '[.[] | select(.type == "required_status_checks")
             | .parameters.strict_required_status_checks_policy] | any(. == true)'

# Who bypasses the ruleset, and how widely. Needs repo admin.
gh api repos/<owner>/<repo>/rulesets --jq '.[].id' \
  | xargs -I{} gh api repos/<owner>/<repo>/rulesets/{} \
      --jq '{name, enforcement, bypass_actors}'
```

Four things to rule out, in order: no `required_status_checks` rule at all;
contexts that match no job name (a required check nothing reports is silently
satisfied on some paths and blocking on others); the legacy protection retired
without the ruleset created; or a rule waiting on a reporter this repo does not
have. Capture the output before changing anything.

Some of this surface is not readable without admin scope — a ruleset's bypass
list, for one. Where that is true, say so here explicitly and name the
compensating control, rather than letting prose stand in for a check that
nobody runs.

### Applying it (ordered — avoids an unprotected window)

Rulesets are not reliably managed by the Probot Settings app, so apply them
out-of-band. **Do the ruleset first, retire legacy protection last.**
Replace `<owner>/<repo>` with the repo slug.

1. **Create the protection ruleset.** `./scripts/bootstrap-repo-settings.sh`
   applies `.github/default-ruleset.json` (the starting-point spec above, as
   `POST`/`PUT` `.../rulesets`) — that is the source of truth for a new repo,
   not this document. Add any bypass entry beyond it by hand (Settings → Rules
   → Rulesets → Main → Bypass list) — including any deploy key or App your
   automation pushes with, which is the one people forget — and record it here
   so `default-ruleset.json` and this document don't drift apart. Add only
   rules whose reporter exists; `required_status_checks` ships with an empty
   context list until real CI job names exist to require. Verify with:

   ```sh
   gh api repos/<owner>/<repo>/rulesets --jq '.[].name'
   ```

2. **Confirm repo-level settings** that the ruleset depends on (already in
   `settings.yml`, but verify they applied):

   ```sh
   gh api repos/<owner>/<repo> \
     --jq '{auto_merge: .allow_auto_merge, delete_branch: .delete_branch_on_merge}'
   # expect: allow_auto_merge true, delete_branch_on_merge true
   ```

3. **Retire the legacy branch protection** only after the ruleset is active and
   verified, so `main` is never unprotected:

   ```sh
   gh api -X DELETE repos/<owner>/<repo>/branches/main/protection
   ```

   The `branches:` block has already been removed from `.github/settings.yml`, so
   the Probot app will not re-create it.

## Actions

| Setting | Value | Why |
|---|---|---|
| Actions permissions | "Allow [org] actions and reusable workflows" + selected third-party | Least privilege. |
| Allowed third-party actions | Pinned to SHA in workflows | Supply-chain hygiene. |
| Workflow permissions (default `GITHUB_TOKEN`) | **Read repository contents and packages permissions** | Per-job scope where needed. |
| Allow GitHub Actions to create and approve PRs | **Off** | Humans approve. |
| Fork PR workflows | "Require approval for first-time contributors" | Stops drive-by token theft. |

## Secrets and security

- **Secret scanning**: On
- **Push protection** (blocks pushes that contain secrets): **On**
- **Dependabot alerts**: On
- **Dependabot security updates**: On
- **Code scanning**: on only if a scanning workflow exists in this repo. A
  code-scanning *ruleset rule* with no workflow to report results blocks every
  PR while every check shows green — add the rule in the same change that adds
  the workflow. Secret scanning plus a strict type checker and linter is a
  defensible posture on its own; say which one you have chosen here.
- **Private vulnerability reporting**: On (so `SECURITY.md` link works)

## Access

- **Default permission for org members**: Read (raise per-team via teams, not org-wide)
- **Outside collaborators**: avoid; prefer adding contractors to a scoped team
- **Two-factor authentication**: required at the org level (verify in org settings)

## Tags and releases

- **Tag protection rule**: protect `v*.*.*` patterns from deletion and force-update
- **Releases**: drafted by humans; generated notes are fine, edit before publish

---

## Applying these settings

### Option 1 — Probot Settings app (recommended)

Install the [Settings app](https://github.com/apps/settings) on the org. The
sibling file `.github/settings.yml` will be applied on every push to `main`.
Note: it covers most but not all of the above (rulesets and some security
settings still need the API or UI).

### Option 2 — `gh` CLI script

See `scripts/bootstrap-repo-settings.sh` for a script that applies the
settings above via the GitHub REST API. Run once per new repo.

### Option 3 — UI

Walk this document top-to-bottom in the repo's Settings tab. Last resort —
prone to drift.

---

## Audit

Quarterly: diff this file against the **live settings**, queried — not against
your memory of them, and not against `settings.yml`, which describes intent
rather than state. Any drift is either a bug in the settings (update them) or a
bug in this file (update the file). Don't let them disagree silently, and don't
record a setting here as on until you have seen it on.
