# Contributing

## Lean quality gates (CI, required on every PR)

`.github/workflows/quality.yml` calls quality-zero-platform's reusable lean
6-gate workflow, pinned to a full commit SHA. For the languages in this
repository (Python):

1. **Lint + format** — ruff (ruff check + ruff format --check)
2. **Types** — basedpyright at `typeCheckingMode: standard` (config in
   `pyproject.toml [tool.basedpyright]`; the shipped server modules
   `index.py`, `engines.py`, `evaluation.py` are the typed surface)
3. **Tests + coverage** — pytest + pytest-cov, **STRICT 100% line AND branch**
   on the declared scope (`[tool.coverage] source = index, engines,
   evaluation`; the stdio entrypoint under `if __name__ == "__main__"` is
   excluded with `# pragma: no cover` as it cannot be unit-tested)
4. **SAST** — opengrep (always on)
5. **Secrets** — gitleaks (always on; `.gitleaks.toml` is a reviewed
   allowlist only)
6. **Deps** — osv-scanner + Dependabot at `deps-severity-floor: any`
   (every finding blocks; `osv-scanner.toml` carries only reviewed, dated,
   reasoned ignores)

CodeQL (`security-and-quality`, python + actions) runs alongside via the
thin caller in `.github/workflows/codeql.yml`.

## Local commands

```bash
python -m pip install -r requirements.txt   # pinned, uv-resolved
python -m pytest                            # full suite
bash scripts/verify                         # the lean gate, exactly as CI runs it
```

`scripts/verify` runs the platform preflight at the exact pinned commit, so
a local pass means what a CI pass means.

## Conventions

- Every change reaches `main` through a pull request; never push to the
  default branch.
- New behavior ships with the tests that pin it; keep coverage at 100/100.
- Security findings are fixed (version bump or code change), never muted;
  an allowlist entry needs a written reason and a date.
