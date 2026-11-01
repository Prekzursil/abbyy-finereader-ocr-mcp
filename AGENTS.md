# AGENTS.md — contributor rules (humans and automation)

These rules apply to every change in this repository, whether made by a
person or by an AI agent acting on their behalf.

1. **Verify before claiming done.** A change is done when CI says so, not
   when it looks right. `bash scripts/verify` runs the same lean gate CI
   runs, at the pinned platform commit.
2. **Never weaken a gate.** Do not lower a coverage threshold, delete a
   lint/type/secret/dependency rule, or add an exclusion without a written,
   dated reason. Only ever make checks stricter or fix the finding.
3. **Tests ship with the change.** New behavior arrives with the tests that
   pin it; the coverage gate measures what the tests actually cover.
4. **Never push to the default branch.** Every change reaches the default
   branch through a pull request, green checks required.
5. **Security findings are fixed, not muted.** Dependabot / CodeQL /
   gitleaks findings are resolved by version bumps or code fixes; an
   allowlist entry needs a review note and a date.
6. **Keep the platform pin current.** `.github/workflows/quality.yml` and
   `codeql.yml` call quality-zero-platform reusable workflows pinned to a
   full 40-hex commit SHA; bump the pin deliberately, never to a branch.
