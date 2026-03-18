import os
import csv
import re
import pandas as pd
from urllib.parse import unquote
from tqdm import tqdm

# ============================================================
# Path Configuration
# ============================================================
log_file_path = 'uploads/access.log'
csv_file_path = 'access.csv'
result_file_path = 'result.txt'
detail_csv_path = 'attack_detection_results.csv'

# ============================================================
# Regex: Apache/Nginx Combined Log Format
# ============================================================
combined_regex = (
    r'^(?P<client>\S+) \S+ (?P<userid>\S+) \[(?P<datetime>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<request>[^ "]+)? HTTP/[0-9.]+" '
    r'(?P<status>[0-9]{3}) (?P<size>[0-9]+|-) '
    r'"(?P<referrer>[^"]*)" "(?P<useragent>[^"]*)"'
)
columns = ['client', 'userid', 'datetime', 'method', 'request', 'status', 'size', 'referrer', 'user_agent']


# ============================================================
# Helper Functions
# ============================================================

def logs_to_df(logfile):
    """Parse access.log into a pandas DataFrame."""
    parsed_lines = []
    with open(logfile) as source_file:
        for line in tqdm(source_file, desc="Parsing log"):
            try:
                log_line = re.findall(combined_regex, line)[0]
                parsed_lines.append(log_line)
            except Exception:
                continue
    return pd.DataFrame(parsed_lines, columns=columns)


def decode_url(encoded_str):
    """Safely decode URL-encoded strings."""
    try:
        return unquote(encoded_str)
    except Exception:
        return encoded_str


def write_error(msg):
    """Write error message to result.txt and exit."""
    with open(result_file_path, 'w') as f:
        f.write(f"Error: {msg}")
    print(f"Error: {msg}")
    exit()


# ============================================================
# Step 1: Parse Log File → CSV
# ============================================================

if not os.path.isfile(log_file_path):
    write_error("access.log file was not found.")

try:
    df = logs_to_df(log_file_path)
    if df.empty:
        write_error("DataFrame is empty, no data processed from log file.")
    df.to_csv(csv_file_path, index=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\', doublequote=True)
    print("Log conversion to CSV successful.")
except Exception as e:
    write_error(f"Log conversion failed: {str(e)}")

if not os.path.isfile(csv_file_path):
    write_error("access.csv file was not created. Please check the log conversion process.")

# ============================================================
# Step 2: Load CSV
# ============================================================

try:
    df = pd.read_csv(csv_file_path)
    print("CSV loaded successfully.")
except Exception as e:
    write_error(f"Error loading CSV: {str(e)}")

# ============================================================
# Step 3: URL Decoding
# ============================================================

df['decoded_request'] = df['request'].apply(decode_url)
df['status'] = df['status'].astype(str)

# ============================================================
# Step 4: Feature Extraction — Attack Detection
# ============================================================

# --- Path Traversal ---
df['is_path_traversal'] = df['decoded_request'].apply(
    lambda x: 1 if re.search(r'(\.\./|\.\.\\|%2F%2E%2E|%2E%2E%2F)', x) else 0
)

# --- SQL Injection ---
sql_patterns = (
    r'(\bUNION\b|\bSELECT\b|\bINSERT\b.*\bINTO\b|\bUPDATE\b.*\bSET\b'
    r'|\bDELETE\b.*\bFROM\b|\bDROP\b.*\bTABLE\b|\bTRUNCATE\b.*\bTABLE\b'
    r'|--|%27|%23|\bAND\b|\bOR\b|\bINFORMATION_SCHEMA\b|\bTABLES\b'
    r'|\bFROM\b|\bWHERE\b|\bEXEC\b|\bCALL\b|\bSHOW\b.*\bTABLES\b'
    r'|\bCHAR\b\(|\bCONVERT\b\(|\bCAST\b\(|\bOR\b\s*1\s*=\s*1|\bAND\b\s*1\s*=\s*1)'
)
df['is_sql_injection'] = df['decoded_request'].apply(
    lambda x: 1 if re.search(sql_patterns, x, re.IGNORECASE) else 0
)

# --- OGNL Injection ---
ognl_patterns = (
    r'(\$\{.*?\}|\#\{|@|\#context|\#request|\#response|\#session'
    r'|\#application|\#servletContext|\#out|\#parameters|\#attr'
    r'|\#header|\#cookie|\#method)'
)
df['is_ognl_injection'] = df['decoded_request'].apply(
    lambda x: 1 if re.search(ognl_patterns, x, re.IGNORECASE) else 0
)

# --- XSS ---
xss_patterns = (
    r'(<script.*?>.*?</script.*?>|%3Cscript.*?%3E.*?%3C/script.*?%3E'
    r'|<.*?on\w+=.*?>|<.*?javascript:.*?>|&#\d+;|&\w+;'
    r'|%3C.*?on\w+%3D.*?%3E|%3C.*?javascript:.*?%3E'
    r'|on\w+\s*=\s*".*?"|on\w+\s*=\s*\'.*?\'|on\w+\s*=\s*[^>]*)'
)
df['is_xss'] = df['decoded_request'].apply(
    lambda x: 1 if re.search(xss_patterns, x, re.IGNORECASE) else 0
)

# --- Remote File Inclusion (RFI) ---
rfi_patterns = r'(https?://[^\s]+(\.php|\.txt|\.html)|ftp://[^\s]+(\.php|\.txt|\.html))'
df['is_rfi'] = df['decoded_request'].apply(
    lambda x: 1 if re.search(rfi_patterns, x, re.IGNORECASE) else 0
)

# --- Malicious Payload ---
malicious_payload_patterns = r'(cmd=|exec=|shell=|bash=|powershell=|cmd\.exe|/bin/bash|/bin/sh|%3B|%26|%7C)'
df['is_malicious_payload'] = df['decoded_request'].apply(
    lambda x: 1 if re.search(malicious_payload_patterns, x, re.IGNORECASE) else 0
)

# --- HTTP Methods Abuse ---
http_methods_abuse_patterns = r'(\bTRACE\b|\bTRACK\b|\bOPTIONS\b|\bCONNECT\b|\bDELETE\b|\bPUT\b|\bPATCH\b)'
df['is_http_methods_abuse'] = df['method'].apply(
    lambda x: 1 if re.search(http_methods_abuse_patterns, x, re.IGNORECASE) else 0
)

# --- Password-Based Attack ---
password_attack_patterns = r'(password=|passwd=|pwd=|login=|username=|user=)'
df['is_password_attack'] = df['decoded_request'].apply(
    lambda x: 1 if re.search(password_attack_patterns, x, re.IGNORECASE) else 0
)

# --- Repeated Login Attempts (per IP, threshold > 5) ---
df['is_repeated_login'] = df.groupby('client')['is_password_attack'].transform('sum').apply(
    lambda x: 1 if x > 5 else 0
)

# --- Unauthorized Access (admin panel keywords) ---
unauthorized_access_patterns = r'(admin|administrator|root|superuser|adminpanel|controlpanel)'
df['is_unauthorized_access'] = df['decoded_request'].apply(
    lambda x: 1 if re.search(unauthorized_access_patterns, x, re.IGNORECASE) else 0
)

# --- DoS Attack (request count per IP, threshold > 100) ---
df['request_count'] = df.groupby('client')['client'].transform('count')
df['is_dos_attack'] = df['request_count'].apply(lambda x: 1 if x > 100 else 0)

# ============================================================
# Step 5: Determine Attack Focus (Priority-based)
# ============================================================

def determine_attack_focus(row):
    if row['is_sql_injection']:
        return 'SQL Injection'
    elif row['is_path_traversal']:
        return 'Path Traversal'
    elif row['is_ognl_injection']:
        return 'OGNL Injection'
    elif row['is_xss']:
        return 'XSS'
    elif row['is_rfi']:
        return 'Remote File Inclusion (RFI)'
    elif row['is_malicious_payload']:
        return 'Malicious Payload'
    elif row['is_http_methods_abuse']:
        return 'HTTP Methods Abuse'
    elif row['is_password_attack']:
        return 'Password-Based Attack'
    elif row['is_unauthorized_access']:
        return 'Unauthorized Access'
    elif row['is_dos_attack']:
        return 'Denial of Service (DoS) Attack'
    else:
        return 'Normal'

df['attack_focus'] = df.apply(determine_attack_focus, axis=1)

# ============================================================
# Step 6: Count Anomalies
# ============================================================

total_path_traversal         = df['is_path_traversal'].sum()
total_sql_injection          = df['is_sql_injection'].sum()
total_ognl_injection         = df['is_ognl_injection'].sum()
total_xss                    = df['is_xss'].sum()
total_rfi                    = df['is_rfi'].sum()
total_malicious_payload      = df['is_malicious_payload'].sum()
total_http_methods_abuse     = df['is_http_methods_abuse'].sum()
total_password_attacks       = df['is_password_attack'].sum()
total_repeated_login         = df['is_repeated_login'].sum()
total_unauthorized_access    = df['is_unauthorized_access'].sum()
total_dos_attacks            = df['is_dos_attack'].sum()

# ============================================================
# Step 7: Save Results
# ============================================================

# Save summary to result.txt
with open(result_file_path, 'w') as result_file:
    result_file.write(f"Total Path Traversal Attacks:        {total_path_traversal}\n")
    result_file.write(f"Total SQL Injection Attacks:         {total_sql_injection}\n")
    result_file.write(f"Total OGNL Injection Attacks:        {total_ognl_injection}\n")
    result_file.write(f"Total XSS Attacks:                   {total_xss}\n")
    result_file.write(f"Total RFI Attacks:                   {total_rfi}\n")
    result_file.write(f"Total Malicious Payload Attacks:     {total_malicious_payload}\n")
    result_file.write(f"Total HTTP Methods Abuse:            {total_http_methods_abuse}\n")
    result_file.write(f"Total Password-Based Attacks:        {total_password_attacks}\n")
    result_file.write(f"Total Repeated Login Attempts:       {total_repeated_login}\n")
    result_file.write(f"Total Unauthorized Access Attempts:  {total_unauthorized_access}\n")
    result_file.write(f"Total DoS Attacks:                   {total_dos_attacks}\n")

# Save full detail to attack_detection_results.csv
df.to_csv(detail_csv_path, index=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\', doublequote=True)

print("Detection completed.")
print(f"Summary saved to '{result_file_path}'.")
print(f"Full detail saved to '{detail_csv_path}'.")
