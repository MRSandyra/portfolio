<?php
declare(strict_types=1);

ini_set('display_errors', '0');
ini_set('log_errors', '1');
mysqli_report(MYSQLI_REPORT_ERROR | MYSQLI_REPORT_STRICT);

try {
    $conn = new mysqli(
        (string) (getenv('DB_HOST') ?: '127.0.0.1'),
        (string) getenv('DB_USER'),
        (string) getenv('DB_PASSWORD'),
        (string) (getenv('DB_NAME') ?: 'anomali_detection'),
        (int) (getenv('DB_PORT') ?: 3306)
    );
    $conn->set_charset('utf8mb4');
} catch (mysqli_sql_exception $exception) {
    error_log('Database connection failed: ' . $exception->getCode());
    http_response_code(500);
    exit('Service is temporarily unavailable.');
}
