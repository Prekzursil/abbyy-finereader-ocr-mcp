## What changed

<!-- One paragraph: the what and the why. -->

## Gate checklist (maps 1:1 to CI)

- [ ] `quality` (lean 6-gate) is green on this PR — lint/format, types, tests+coverage, SAST, secrets, deps
- [ ] Tests: new/changed behavior is pinned by tests; coverage stays at the declared 100% line+branch scope (or the scope change is declared in CONTRIBUTING.md with a reason)
- [ ] No new findings were muted: any new ignore/allowlist entry carries a dated, written reason
- [ ] `CodeQL` is green (no new alert of any severity)
- [ ] Docs/changelog updated if user-visible

## Notes for reviewers

<!-- Anything non-obvious, follow-ups, or deliberate trade-offs. -->
