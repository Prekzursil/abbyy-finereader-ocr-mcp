# Security Policy

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-reviewing/privately-reporting-a-security-vulnerability)
on this repository (**Security → Report a vulnerability**). Do not open a
public issue for anything security-sensitive.

Include the engine involved (RapidOCR, Tesseract, FineReader), the tool call
that triggers the issue, and a proof of concept.

## Scope

The MCP server code in this repository is in scope. The bundled OCR engine
binaries and third-party models are reported to their upstream projects;
this repository coordinates the upgrade path here.

## Hardening notes

- File-access tools sandbox through `OCR_MCP_ALLOWED_DIRS` (os.pathsep-separated
  allowlist). Unset = the process may read any local file it can access — the
  documented trusted-local-client mode. Set the variable to constrain every
  tool path to the listed directories; every path is resolved and checked at a
  single choke point (`_safe_path`) before any engine sees it.
- Dependabot security updates are enabled; CodeQL (security-and-quality) and
  the lean quality gate (gitleaks, opengrep, osv-scanner at zero tolerance)
  run on every pull request.
