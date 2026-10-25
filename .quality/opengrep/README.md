# Curated SAST ruleset (gate 4)

These rules were seeded from
[quality-zero-platform's curated ruleset](https://github.com/Prekzursil/quality-zero-platform/tree/main/.quality/opengrep)
by the fleet baseline (`docs/FLEET-BASELINE.md` in quality-zero-platform). The
lean gate scans `Prekzursil/abbyy-finereader-ocr-mcp` with exactly these rules on every change and
passes only on **zero findings**: no baseline file, no `--config auto`, no
network-fetched rule packs.

- Add rules for this repository's own risks here. Keep every rule high-signal
  and reviewed; never delete or narrow a rule to make a finding go away.
- Suppress a genuine false positive inline with a greppable
  `# nosemgrep: <rule-id> -- <reason>` on the first line of the matched
  construct.
- `.semgrepignore` at the repository root decides the scanned set. Never
  exclude first-party tests: without that file opengrep silently skips
  `test/`, `tests/` and every `*_test.go`.
