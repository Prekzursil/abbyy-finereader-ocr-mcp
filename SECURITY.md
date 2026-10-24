# Security Policy

## Supported versions

Only the latest commit on `main` receives security fixes.

| Version | Supported |
| ------- | --------- |
| `main` | ✅ |
| older commits, tags and branches | ❌ update first |

## Reporting a vulnerability

**Please use [private vulnerability reporting](https://github.com/Prekzursil/abbyy-finereader-ocr-mcp/security/advisories/new)** — the *Report a vulnerability* button in this repository's *Security* tab. Reports there go privately to the maintainer and allow coordinated disclosure.

- Do **not** open a public issue, pull request or discussion for anything
  security-sensitive, and keep details out of commit messages.
- Include reproduction steps, the affected file, module or workflow, and the
  exact commit you tested.
- You should get an acknowledgement within **72 hours** and a fix or
  mitigation plan within **14 days**, workload permitting.
- Please allow coordinated disclosure — until a fix ships or 90 days have
  passed — before publishing details.

## Scope

In scope: anything in this repository that could leak data, corrupt data, or
execute unintended code when it is used as documented — including its GitHub
Actions workflows and its dependency chain.

Out of scope: findings that require an already-compromised machine or account,
denial of service by deliberately exhausting local resources, and purely
cosmetic issues.

## How this repository is kept secure

This repository is governed by
[quality-zero-platform](https://github.com/Prekzursil/quality-zero-platform)
(QZP). Its policy is **zero tolerance**: a finding of *any* severity fails CI —
there is no "high and above only" floor.

- **CodeQL** — GitHub default setup with the *extended* query suite and the remote + local threat model, on every pull request and weekly; a pull request that introduces a code-scanning alert of any severity cannot be merged.
- **SAST** — opengrep on every change.
- **Secrets** — gitleaks on every change; GitHub secret scanning with push protection is enabled.
- **Dependencies** — osv-scanner on every lockfile and manifest, plus Dependabot alerts and grouped security updates.

Report suspected dependency or supply-chain compromises through the same
private channel, and via the affected package's own advisory process.
