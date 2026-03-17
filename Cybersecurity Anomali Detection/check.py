import os
import pandas as pd
from urllib.parse import unquote
import re
import csv
from tqdm import tqdm

# Path for uploaded log file and converted CSV file
log_file_path = 'uploads/access.log'
csv_file_path = 'access.csv'

# Define regex patterns and columns
combined_regex = r'^(?P<client>\S+) \S+ (?P<userid>\S+) \[(?P<datetime>[^\]]+)\] "(?P<method>[A-Z]+) (?P<request>[^ "]+)? HTTP/[0-9.]+" (?P<status>[0-9]{3}) (?P<size>[0-9]+|-) "(?P<referrer>[^"]*)" "(?P<useragent>[^"]*)"'
columns = ['client', 'userid', 'datetime', 'method', 'request', 'status', 'size', 'referrer', 'user_agent']

def logs_to_df(logfile):
    parsed_lines = []
    with open(logfile) as source_file:
        for line in tqdm(source_file):
            try:
                log_line = re.findall(combined_regex, line)[0]
                parsed_lines.append(log_line)
            except Exception as e:
                continue

    # Convert parsed lines to DataFrame
    df = pd.DataFrame(parsed_lines, columns=columns)
    return df

def decode_url(encoded_str):
    try:
        return unquote(encoded_str)
    except Exception as e:
        return encoded_str

# Ensure the log file exists
if not os.path.isfile(log_file_path):
    with open('result.txt', 'w') as result_file:
        result_file.write("Error: access.log file was not found.")
    print("Error: access.log file was not found.")
    exit()

# Convert log file to DataFrame
try:
    df = logs_to_df(log_file_path)
    
    # Debugging: Check if DataFrame is empty
    if df.empty:
        with open('result.txt', 'w') as result_file:
            result_file.write("Error: DataFrame is empty, no data processed from log file.")
        print("Error: DataFrame is empty.")
        exit()

    # Save to CSV
    df.to_csv(csv_file_path, index=False)
    print("Log conversion to CSV successful.")
except Exception as e:
    with open('result.txt', 'w') as result_file:
        result_file.write(f"Error during log conversion: {str(e)}")
    print(f"Error during log conversion: {str(e)}")
    exit()

# Check if CSV file was created
if not os.path.isfile(csv_file_path):
    with open('result.txt', 'w') as result_file:
        result_file.write("Error: access.csv file was not created. Please check the log conversion process.")
    print("Error: access.csv file was not created. Please check the log conversion process.")
    exit()

# Load dataset
try:
    df = pd.read_csv(csv_file_path)
    print("CSV loaded successfully.")
except Exception as e:
    with open('result.txt', 'w') as result_file:
        result_file.write(f"Error loading CSV: {str(e)}")
    print(f"Error loading CSV: {str(e)}")
    exit()

# URL Decoding
df['decoded_request'] = df['request'].apply(decode_url)

# Feature Extraction for Path Traversal
df['is_path_traversal'] = df['decoded_request'].apply(lambda x: 1 if re.search(r'(\.\./|\.\.\\|%2F%2E2E|%2E2E%2F)', x) else 0)

# Feature Extraction for SQL Injection
sql_patterns = r"(\bUNION\b|\bSELECT\b|--|%27|%23|\bAND\b|\bOR\b|\bINFORMATION_SCHEMA\b|\bTABLES\b|\bFROM\b|\bWHERE\b)"
df['is_sql_injection'] = df['decoded_request'].apply(lambda x: 1 if re.search(sql_patterns, x, re.IGNORECASE) else 0)

# Feature Extraction for OGNL Injection
ognl_patterns = r"(\$\{.*?\}|\#\{|@|\#context|\#request|\#response|\#session|\#application|\#servletContext|\#out|\#parameters|\#attr|\#header|\#cookie|\#method)"
df['is_ognl_injection'] = df['decoded_request'].apply(lambda x: 1 if re.search(ognl_patterns, x, re.IGNORECASE) else 0)

# Feature Extraction for XSS
xss_patterns = r"(<script.*?>.*?</script.*?>|%3Cscript.*?%3E.*?%3C/script.*?%3E|<.*?on\w+=.*?>|<.*?javascript:.*?>|%3C.*?on\w+%3D.*?%3E|%3C.*?javascript:.*?%3E)"
df['is_xss'] = df['decoded_request'].apply(lambda x: 1 if re.search(xss_patterns, x, re.IGNORECASE) else 0)

# Feature Extraction for Remote File Inclusion (RFI)
rfi_patterns = r"(https?://[^\s]+(\.php|\.txt|\.html)|ftp://[^\s]+(\.php|\.txt|\.html))"
df['is_rfi'] = df['decoded_request'].apply(lambda x: 1 if re.search(rfi_patterns, x, re.IGNORECASE) else 0)

# Feature Extraction for Malicious Payloads
malicious_payload_patterns = r"(cmd=|exec=|shell=|bash=|powershell=|cmd.exe|/bin/bash|/bin/sh|%3B|%26|%7C)"
df['is_malicious_payload'] = df['decoded_request'].apply(lambda x: 1 if re.search(malicious_payload_patterns, x, re.IGNORECASE) else 0)

# Feature Extraction for HTTP Methods Abuse
http_methods_abuse_patterns = r"(\bTRACE\b|\bTRACK\b|\bOPTIONS\b|\bCONNECT\b|\bDELETE\b|\bPUT\b|\bPATCH\b)"
df['is_http_methods_abuse'] = df['method'].apply(lambda x: 1 if re.search(http_methods_abuse_patterns, x, re.IGNORECASE) else 0)

# Feature Extraction for Password-Based Attacks
password_attack_patterns = r"(password=|passwd=|pwd=|login=|username=|user=)"
df['is_password_attack'] = df['decoded_request'].apply(lambda x: 1 if re.search(password_attack_patterns, x, re.IGNORECASE) else 0)

# Identify Repeated Login Attempts by IP
df['is_repeated_login'] = df.groupby('client')['is_password_attack'].transform('sum').apply(lambda x: 1 if x > 5 else 0)

# Determine attack focus based on first detected attack (including Password-Based Attacks)
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
    else:
        return 'Normal'

df['attack_focus'] = df.apply(determine_attack_focus, axis=1)

# Count anomalies
total_path_traversal = df[df['is_path_traversal'] == 1].shape[0]
total_sql_injection = df[df['is_sql_injection'] == 1].shape[0]
total_ognl_injection = df[df['is_ognl_injection'] == 1].shape[0]
total_xss = df[df['is_xss'] == 1].shape[0]
total_rfi = df[df['is_rfi'] == 1].shape[0]
total_malicious_payload = df[df['is_malicious_payload'] == 1].shape[0]
total_http_methods_abuse = df[df['is_http_methods_abuse'] == 1].shape[0]
total_password_attacks = df[df['is_password_attack'] == 1].shape[0]
total_repeated_login_attempts = df[df['is_repeated_login'] == 1].shape[0]

# Save the results to a text file
with open('result.txt', 'w') as result_file:
    result_file.write(f"Total Path Traversal Attacks: {total_path_traversal}\n")
    result_file.write(f"Total SQL Injection Attacks: {total_sql_injection}\n")
    result_file.write(f"Total OGNL Injection Attacks: {total_ognl_injection}\n")
    result_file.write(f"Total XSS Attacks: {total_xss}\n")
    result_file.write(f"Total Remote File Inclusion (RFI) Attacks: {total_rfi}\n")
    result_file.write(f"Total Malicious Payload Attacks: {total_malicious_payload}\n")
    result_file.write(f"Total HTTP Methods Abuse: {total_http_methods_abuse}\n")
    result_file.write(f"Total Password-Based Attacks: {total_password_attacks}\n")
    result_file.write(f"Total Repeated Login Attempts: {total_repeated_login_attempts}\n")

print("Detection completed. Results saved to 'result.txt'.")
