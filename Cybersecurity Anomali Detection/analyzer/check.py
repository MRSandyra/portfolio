#!/usr/bin/env python3
"""Analyze one Apache/Nginx combined log into an isolated output directory."""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

MAX_INPUT_BYTES = 5 * 1024 * 1024
COLUMNS = ['client', 'userid', 'datetime', 'method', 'request', 'status', 'size', 'referrer', 'user_agent']
COMBINED_REGEX = re.compile(
    r'^(?P<client>\S+) \S+ (?P<userid>\S+) \[(?P<datetime>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<request>[^ "]+)? HTTP/[0-9.]+" '
    r'(?P<status>[0-9]{3}) (?P<size>[0-9]+|-) '
    r'"(?P<referrer>[^"]*)" "(?P<useragent>[^"]*)"'
)
PATTERNS = {
    'is_path_traversal': r'(\.\./|\.\.\\|%2F%2E%2E|%2E%2E%2F)',
    'is_sql_injection': r'(\bUNION\b|\bSELECT\b|\bINSERT\b.*\bINTO\b|\bUPDATE\b.*\bSET\b|\bDELETE\b.*\bFROM\b|\bDROP\b.*\bTABLE\b|\bTRUNCATE\b.*\bTABLE\b|--|%27|%23|\bAND\b|\bOR\b|\bINFORMATION_SCHEMA\b|\bTABLES\b|\bFROM\b|\bWHERE\b|\bEXEC\b|\bCALL\b|\bSHOW\b.*\bTABLES\b|\bCHAR\b\(|\bCONVERT\b\(|\bCAST\b\(|\bOR\b\s*1\s*=\s*1|\bAND\b\s*1\s*=\s*1)',
    'is_ognl_injection': r'(\$\{.*?\}|\#\{|@|\#context|\#request|\#response|\#session|\#application|\#servletContext|\#out|\#parameters|\#attr|\#header|\#cookie|\#method)',
    'is_xss': r'(<script.*?>.*?</script.*?>|%3Cscript.*?%3E.*?%3C/script.*?%3E|<.*?on\w+=.*?>|<.*?javascript:.*?>|&#\d+;|&\w+;|%3C.*?on\w+%3D.*?%3E|%3C.*?javascript:.*%3E|on\w+\s*=\s*".*?"|on\w+\s*=\s*\'.*?\'|on\w+\s*=\s*[^>]*)',
    'is_rfi': r'(https?://[^\s]+(\.php|\.txt|\.html)|ftp://[^\s]+(\.php|\.txt|\.html))',
    'is_malicious_payload': r'(cmd=|exec=|shell=|bash=|powershell=|cmd\.exe|/bin/bash|/bin/sh|%3B|%26|%7C)',
    'is_unauthorized_access': r'(admin|administrator|root|superuser|adminpanel|controlpanel)',
}
COMPILED_PATTERNS = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PATTERNS.items()}
HTTP_METHOD_ABUSE = re.compile(r'(\bTRACE\b|\bTRACK\b|\bOPTIONS\b|\bCONNECT\b|\bDELETE\b|\bPUT\b|\bPATCH\b)', re.IGNORECASE)
PASSWORD_PATTERN = re.compile(r'(password=|passwd=|pwd=|login=|username=|user=)', re.IGNORECASE)
PRIORITY = [
    ('is_sql_injection', 'SQL Injection'), ('is_path_traversal', 'Path Traversal'),
    ('is_ognl_injection', 'OGNL Injection'), ('is_xss', 'XSS'), ('is_rfi', 'Remote File Inclusion (RFI)'),
    ('is_malicious_payload', 'Malicious Payload'), ('is_http_methods_abuse', 'HTTP Methods Abuse'),
    ('is_password_attack', 'Password-Based Attack'), ('is_unauthorized_access', 'Unauthorized Access'),
    ('is_dos_attack', 'Denial of Service (DoS) Attack'),
]
SUMMARY = [
    ('Total Path Traversal Attacks', 'is_path_traversal'), ('Total SQL Injection Attacks', 'is_sql_injection'),
    ('Total OGNL Injection Attacks', 'is_ognl_injection'), ('Total XSS Attacks', 'is_xss'), ('Total RFI Attacks', 'is_rfi'),
    ('Total Malicious Payload Attacks', 'is_malicious_payload'), ('Total HTTP Methods Abuse', 'is_http_methods_abuse'),
    ('Total Password-Based Attacks', 'is_password_attack'), ('Total Repeated Login Attempts', 'is_repeated_login'),
    ('Total Unauthorized Access Attempts', 'is_unauthorized_access'), ('Total DoS Attacks', 'is_dos_attack'),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze an Apache/Nginx combined access log.')
    parser.add_argument('log_file', type=Path, help='uploaded access-log file')
    parser.add_argument('--output-dir', type=Path, required=True, help='private workspace for generated files')
    return parser.parse_args()


def validate_input(log_file):
    if not log_file.is_file():
        raise ValueError('Log file was not found.')
    size = log_file.stat().st_size
    if size < 1 or size > MAX_INPUT_BYTES:
        raise ValueError('Log file must be between 1 byte and 5 MB.')
    with log_file.open('rb') as source:
        if b'\0' in source.read(16384):
            raise ValueError('Log file must be plain text.')


def logs_to_rows(logfile):
    rows = []
    with logfile.open(encoding='utf-8', errors='replace') as source_file:
        for line in source_file:
            match = COMBINED_REGEX.match(line)
            if match:
                rows.append(dict(zip(COLUMNS, match.groups())))
    return rows


def analyze(log_file, output_dir):
    validate_input(log_file)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    rows = logs_to_rows(log_file)
    if not rows:
        raise ValueError('No valid combined-log entries were found.')
    for row in rows:
        request = unquote(row['request'] or '')
        row['decoded_request'] = request
        for name, pattern in COMPILED_PATTERNS.items():
            row[name] = int(bool(pattern.search(request)))
        row['is_http_methods_abuse'] = int(bool(HTTP_METHOD_ABUSE.search(row['method'])))
        row['is_password_attack'] = int(bool(PASSWORD_PATTERN.search(request)))
    login_attempts = Counter(row['client'] for row in rows if row['is_password_attack'])
    request_counts = Counter(row['client'] for row in rows)
    for row in rows:
        row['is_repeated_login'] = int(login_attempts[row['client']] > 5)
        row['request_count'] = request_counts[row['client']]
        row['is_dos_attack'] = int(row['request_count'] > 100)
        row['attack_focus'] = 'Normal'
        for field, label in reversed(PRIORITY):
            if row[field]:
                row['attack_focus'] = label
    generated_columns = ['decoded_request', *PATTERNS.keys(), 'is_http_methods_abuse', 'is_password_attack', 'is_repeated_login', 'request_count', 'is_dos_attack', 'attack_focus']
    with (output_dir / 'access.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader(); writer.writerows([{field: row[field] for field in COLUMNS} for row in rows])
    with (output_dir / 'result.txt').open('w', encoding='utf-8') as output:
        for label, field in SUMMARY:
            output.write(f'{label}: {sum(row[field] for row in rows)}\n')
    with (output_dir / 'attack_detection_results.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=[*COLUMNS, *generated_columns], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args()
    try:
        analyze(args.log_file.resolve(), args.output_dir.resolve())
    except (OSError, ValueError) as error:
        print(f'Analysis failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
