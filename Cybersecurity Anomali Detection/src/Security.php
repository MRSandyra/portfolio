<?php
declare(strict_types=1);

const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
const RESET_TOKEN_TTL_SECONDS = 3600;

function e(?string $value): string
{
    return htmlspecialchars((string) $value, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function start_secure_session(): void
{
    if (session_status() === PHP_SESSION_NONE) {
        session_set_cookie_params([
            'httponly' => true,
            'secure' => (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off'),
            'samesite' => 'Lax',
        ]);
        session_start();
    }
}

function csrf_token(): string
{
    start_secure_session();
    if (empty($_SESSION['csrf_token'])) {
        $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
    }
    return $_SESSION['csrf_token'];
}

function verify_csrf(): bool
{
    start_secure_session();
    return isset($_POST['csrf_token'], $_SESSION['csrf_token'])
        && is_string($_POST['csrf_token'])
        && hash_equals($_SESSION['csrf_token'], $_POST['csrf_token']);
}

function security_headers(): void
{
    header('X-Content-Type-Options: nosniff');
    header('X-Frame-Options: DENY');
    header('Referrer-Policy: no-referrer');
    header("Content-Security-Policy: default-src 'self'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'; style-src 'self'; script-src 'self'");
}

function password_error(string $password): ?string
{
    if (strlen($password) < 12) {
        return 'Password must contain at least 12 characters.';
    }
    if (strlen($password) > 4096) {
        return 'Password is too long.';
    }
    return null;
}

function safe_base_url(): ?string
{
    $url = rtrim((string) getenv('APP_BASE_URL'), '/');
    return filter_var($url, FILTER_VALIDATE_URL) && str_starts_with($url, 'https://') ? $url : null;
}
