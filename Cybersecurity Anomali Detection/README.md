# Cybersecurity Anomaly Detection

A PHP and Python portfolio project that analyzes Apache/Nginx Combined Log Format files for suspicious web requests. It detects SQL injection, path traversal, OGNL, XSS, RFI, shell-like payloads, HTTP method abuse, password-related attacks, unauthorized-path probing, repeated logins, and high-volume requests.

## Security posture

- Database configuration is supplied exclusively through environment variables; copy `.env.example` for local reference but do not commit a populated `.env`.
- Password-reset tokens are 256-bit random, URL-safe values. Only their SHA-256 digest is stored, they expire after one hour, and the password update consumes the token atomically.
- Reset requests always return the same message, whether or not the account exists. Reset links use `APP_BASE_URL` and are URL encoded.
- The reset flow has no token, query, account-ID, or database-error debug output. Existing plaintext reset tokens are invalidated by the migration.
- Uploads are limited to 5 MB, validated by extension, MIME, NUL-byte check, and Combined Log Format content check.
- Each analysis gets a random private workspace outside the web root. The Python program receives explicit input and output paths; there are no shared `result.txt`, `access.csv`, or CSV outputs.
- The analyzer starts using `proc_open()` with an argument array and `bypass_shell`, has a configurable timeout, and receives no constructed shell command.
- Result CSV downloads are restricted to the session that created the analysis and workspaces are safely purged after one hour.
- Dynamic HTML is escaped, forms use CSRF tokens, sessions use HttpOnly/SameSite cookies, and response headers reduce MIME-sniffing, framing, and reset-link referrer exposure.

## Requirements

- PHP 8.0+ with `mysqli`, `fileinfo`, and `proc_open` enabled
- MySQL or MariaDB
- Python 3.9+ (standard library only)
- A working server-side mail transport for PHP `mail()` (or replace that delivery adapter in production)

## Setup

1. Import the clean schema: `mysql -u USER -p anomali_detection < anomali_detection.sql`.
2. For an existing database, first run `migrations/20261120_harden_reset_tokens.sql`. It preserves the existing reset columns, adds a lookup index, and invalidates legacy plaintext tokens.
3. Set the environment variables in `.env.example` in your web-server configuration. This application deliberately does not load `.env` files itself, keeping runtime secrets outside the document root and source control.
4. Set `ANALYSIS_WORKSPACE_ROOT` to an absolute directory outside the web root that is writable only by the web-server account. If it is unset, the application uses the operating system temporary directory with restrictive permissions.

`APP_BASE_URL` must be an HTTPS public URL without a trailing slash. The reset handler will not issue a link if it is absent or not HTTPS. Set `RESET_EMAIL_FROM` to an allowed sender. Failure to send a message clears the just-issued token and still shows the generic response.

## Use

Open `index.php`, upload a `.log` or `.txt` file in Apache/Nginx Combined Log Format, and review the escaped summary. The detailed CSV remains available to that browser session for one hour. For command-line use:

```bash
python check.py path/to/access.log --output-dir path/to/private-job
```

The output directory receives `access.csv`, `result.txt`, and `attack_detection_results.csv`; do not use a shared directory for concurrent jobs.

## Tests

```bash
php tests/run.php
python -m unittest tests/test_analyzer_concurrency.py
```

The PHP test covers reset-token format, hashing, expiry, modeled one-time consumption, and upload extension/content validation. The Python integration test runs two analyses concurrently and asserts that their output files do not collide.

## Threat model and remaining limitations

The main attackers considered are unauthenticated users enumerating reset accounts or replaying reset links; authenticated or anonymous users uploading malicious/binary/oversized files; concurrent users racing output files; and users injecting rendered log data. The controls above reduce those attack paths, but they do not replace web-server request limits, TLS, rate limiting, monitoring, mail-provider hardening, malware scanning, or sandbox/container isolation for Python in a high-risk deployment.

Detection is intentionally regex-based. It can produce false positives from benign queries and false negatives from novel encodings, multi-stage payloads, non-Combined log formats, encrypted traffic, and attacks that do not appear in the request path. The DoS and repeated-login thresholds are simple whole-file heuristics rather than time-windowed behavioral analytics. Treat the tool as triage support, not an IDS decision engine.
