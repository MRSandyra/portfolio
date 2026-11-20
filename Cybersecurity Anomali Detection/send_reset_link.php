<?php
declare(strict_types=1);

require_once __DIR__ . '/security.php';
require_once __DIR__ . '/ResetToken.php';
require_once __DIR__ . '/koneksi.php';

start_secure_session();
security_headers();
if ($_SERVER['REQUEST_METHOD'] !== 'POST' || !verify_csrf()) {
    http_response_code(400);
    exit('Invalid request.');
}

$email = filter_var(trim((string) ($_POST['email'] ?? '')), FILTER_VALIDATE_EMAIL);
$genericMessage = 'If that email address is registered, a password-reset link will be sent shortly.';

if ($email !== false && ($baseUrl = safe_base_url()) !== null) {
    try {
        $tokenData = ResetToken::issue();
        $stmt = $conn->prepare('UPDATE login SET reset_token = ?, reset_token_expiry = ? WHERE email = ?');
        $stmt->bind_param('sss', $tokenData['hash'], $tokenData['expiresAt'], $email);
        $stmt->execute();

        if ($stmt->affected_rows === 1) {
            $link = $baseUrl . '/reset_password.php?token=' . rawurlencode($tokenData['token']);
            $from = (string) getenv('RESET_EMAIL_FROM');
            $headers = $from !== '' && filter_var($from, FILTER_VALIDATE_EMAIL)
                ? "From: {$from}\r\nContent-Type: text/plain; charset=UTF-8"
                : 'Content-Type: text/plain; charset=UTF-8';
            $sent = @mail($email, 'Password reset request', "Use this one-time link within one hour:\n{$link}\n\nIf you did not request it, you can ignore this email.", $headers);
            if (!$sent) {
                $clear = $conn->prepare('UPDATE login SET reset_token = NULL, reset_token_expiry = NULL WHERE email = ? AND reset_token = ?');
                $clear->bind_param('ss', $email, $tokenData['hash']);
                $clear->execute();
                error_log('Password reset delivery failed.');
            }
        }
    } catch (Throwable $exception) {
        error_log('Password reset request failed: ' . $exception->getCode());
    }
}

$_SESSION['reset_request_message'] = $genericMessage;
header('Location: Reset_form.php', true, 303);
exit;
