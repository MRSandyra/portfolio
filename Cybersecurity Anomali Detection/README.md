# 🛡️ Anomaly Detection System

A web-based security monitoring application that analyzes Apache/Nginx access logs to detect various types of web attacks and anomalous behavior in real time.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Attack Detection Logic](#attack-detection-logic)
- [Output Files](#output-files)
- [Web Interface](#web-interface)
- [Database](#database)

---

## Overview

The **Anomaly Detection System** is a hybrid PHP + Python application designed to parse web server access logs, identify malicious traffic patterns, and display the results through a web dashboard. The Python script (`check.py`) handles the core log analysis and attack detection, while the PHP files provide a user-facing web interface with authentication.

---

## ✨ Features

- Parses Apache/Nginx Combined Log Format access logs
- Detects **11 types** of web attacks and anomalies
- URL decoding for obfuscated attack payloads
- Per-IP request counting for DoS detection
- Brute-force login attempt detection
- Exports results to CSV for further analysis
- Web dashboard with user authentication, registration, and password reset

---

## 📁 Project Structure

```
anomali_detection/
├── check.py                        # Core Python script: log parsing & attack detection
├── index.php                       # Main dashboard page
├── result.php                      # Detection results display page
├── login.php                       # User login page
├── register.php                    # User registration page
├── logout.php                      # Session logout handler
├── koneksi.php                     # Database connection configuration
├── account_settings.php            # User account settings
├── Reset_form.php                  # Password reset form
├── reset_password.html             # Password reset HTML page
├── reset_password.php              # Password reset handler
├── reset2.php                      # Secondary reset handler
├── send_reset_link.php             # Sends password reset link via email
├── update_password.php             # Updates password in database
├── anomali_detection.sql           # Database schema and initial data
├── attack_detection_results.csv    # Output: full detection results per log entry
├── package.json                    # Node.js dependencies (if any)
├── package-lock.json               # Node.js lock file
├── log.txt                         # Application log
├── send_success_log.txt            # Email send success log
├── php_command_output_log.txt      # PHP command execution log
└── README.md                       # Project documentation
```

> **Note:** `check.py` expects an `uploads/access.log` file at runtime and produces `access.csv`, `result.txt`, and `attack_detection_results.csv` as output.

---

## ⚙️ Requirements

### Python

- Python 3.7+
- Required packages:

```
pandas
tqdm
```

Install with:

```bash
pip install pandas tqdm
```

### PHP & Web Server

- PHP 7.4+ or 8.x
- MySQL / MariaDB
- Apache or Nginx web server
- PHPMailer (or similar) for password reset email functionality

---

## 🚀 Installation

1. **Clone or copy** the project to your web server's document root:

```bash
cp -r anomali_detection/ /var/www/html/anomali_detection/
```

2. **Import the database schema:**

```bash
mysql -u root -p < anomali_detection.sql
```

3. **Configure the database connection** in `koneksi.php`:

```php
$host = 'localhost';
$db   = 'anomali_detection';
$user = 'your_db_user';
$pass = 'your_db_password';
```

4. **Install Python dependencies:**

```bash
pip install pandas tqdm
```

5. **Set permissions** for the upload directory:

```bash
mkdir -p uploads/
chmod 755 uploads/
```

6. Access the application at `http://localhost/anomali_detection/`.

---

## 🔧 Configuration

The following paths are configured at the top of `check.py` and can be adjusted as needed:

| Variable          | Default                          | Description                        |
|-------------------|----------------------------------|------------------------------------|
| `log_file_path`   | `uploads/access.log`             | Input: web server access log file  |
| `csv_file_path`   | `access.csv`                     | Intermediate: parsed log as CSV    |
| `result_file_path`| `result.txt`                     | Output: attack count summary       |
| `detail_csv_path` | `attack_detection_results.csv`   | Output: full per-request results   |

---

## 🖥️ Usage

### Running the Detection Script

1. Place your Apache/Nginx access log file at `uploads/access.log`.

2. Run the detection script:

```bash
python check.py
```

3. Review the summary output in `result.txt` and the full detailed results in `attack_detection_results.csv`.

### Example Summary Output (`result.txt`)

```
Total Path Traversal Attacks:        12
Total SQL Injection Attacks:         47
Total OGNL Injection Attacks:        3
Total XSS Attacks:                   19
Total RFI Attacks:                   2
Total Malicious Payload Attacks:     8
Total HTTP Methods Abuse:            5
Total Password-Based Attacks:        31
Total Repeated Login Attempts:       14
Total Unauthorized Access Attempts:  62
Total DoS Attacks:                   200
```

---

## 🔍 Attack Detection Logic

`check.py` applies regex-based pattern matching on each decoded URL request. Detection categories and their priority order:

| Priority | Attack Type                  | Detection Method                                         |
|----------|------------------------------|----------------------------------------------------------|
| 1        | **SQL Injection**            | Keywords: `UNION`, `SELECT`, `DROP`, `--`, `%27`, etc.  |
| 2        | **Path Traversal**           | Patterns: `../`, `..\`, `%2F%2E%2E`, etc.               |
| 3        | **OGNL Injection**           | Patterns: `${...}`, `#context`, `#request`, etc.        |
| 4        | **XSS**                      | Patterns: `<script>`, `onXXX=`, `javascript:`, etc.     |
| 5        | **Remote File Inclusion**    | External URLs ending in `.php`, `.txt`, `.html`         |
| 6        | **Malicious Payload**        | `cmd=`, `exec=`, `/bin/bash`, `powershell=`, etc.       |
| 7        | **HTTP Methods Abuse**       | Methods: `TRACE`, `TRACK`, `OPTIONS`, `CONNECT`, etc.   |
| 8        | **Password-Based Attack**    | Params: `password=`, `passwd=`, `login=`, `user=`       |
| 9        | **Unauthorized Access**      | Paths: `admin`, `administrator`, `root`, `superuser`    |
| 10       | **Denial of Service (DoS)**  | > 100 requests from a single IP address                 |
| –        | **Repeated Login Attempts**  | > 5 password-related requests from a single IP          |

> When multiple attack types are detected in a single request, the **highest-priority** category is assigned as the `attack_focus`.

---

## 📄 Output Files

### `result.txt`
A plain-text summary of total detected attacks per category. Suitable for quick review or integration with alerting systems.

### `attack_detection_results.csv`
A full per-request CSV export containing all original log fields plus the following detection columns:

| Column                   | Description                                      |
|--------------------------|--------------------------------------------------|
| `decoded_request`        | URL-decoded version of the request path          |
| `is_path_traversal`      | 1 if path traversal detected, else 0             |
| `is_sql_injection`       | 1 if SQL injection detected, else 0              |
| `is_ognl_injection`      | 1 if OGNL injection detected, else 0             |
| `is_xss`                 | 1 if XSS detected, else 0                        |
| `is_rfi`                 | 1 if remote file inclusion detected, else 0      |
| `is_malicious_payload`   | 1 if malicious payload detected, else 0          |
| `is_http_methods_abuse`  | 1 if HTTP method abuse detected, else 0          |
| `is_password_attack`     | 1 if password-based attack detected, else 0      |
| `is_repeated_login`      | 1 if IP has >5 password attempts, else 0         |
| `is_unauthorized_access` | 1 if admin panel access attempt detected, else 0 |
| `request_count`          | Total requests from this IP address              |
| `is_dos_attack`          | 1 if IP has >100 requests, else 0                |
| `attack_focus`           | Highest-priority attack type label               |

---

## 🌐 Web Interface

The PHP-based web interface provides:

- **Login / Register** — Secure user authentication (`login.php`, `register.php`)
- **Dashboard** — Overview of detection activity (`index.php`)
- **Results View** — Paginated display of `attack_detection_results.csv` data (`result.php`)
- **Account Settings** — Manage user profile (`account_settings.php`)
- **Password Reset** — Email-based password recovery flow (`Reset_form.php`, `send_reset_link.php`, `reset_password.php`, `update_password.php`)

---

## 🗄️ Database

The application uses a MySQL database. Import the schema using:

```bash
mysql -u root -p anomali_detection < anomali_detection.sql
```

The database stores user accounts, session data, and optionally detection results for display in the web dashboard.

---

## 📝 Notes

- The log parser follows the **Apache/Nginx Combined Log Format**. Custom log formats may require modifying the `combined_regex` pattern in `check.py`.
- All URL-encoded attack payloads are decoded before pattern matching to improve detection accuracy.
- DoS and repeated login detection thresholds (100 requests and 5 login attempts respectively) can be adjusted directly in `check.py`.
- This tool is intended for **defensive security monitoring** purposes only.

---

## 📄 License

This project is proprietary. All rights reserved.
