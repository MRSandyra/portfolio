<?php
declare(strict_types=1);

require_once dirname(__DIR__) . '/src/Security.php';
require_once dirname(__DIR__) . '/src/ResetToken.php';
require_once dirname(__DIR__) . '/config/database.php';

start_secure_session();
security_headers();
$message = null;
$token = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $token = (string) ($_POST['token'] ?? '');
    $password = (string) ($_POST['password'] ?? '');
    $confirmation = (string) ($_POST['password_confirmation'] ?? '');
    if (!verify_csrf()) {
        $message = 'Your session expired. Please open the reset link again.';
    } elseif (!ResetToken::isWellFormed($token)) {
        $message = 'This reset link is invalid or has expired.';
    } elseif ($password !== $confirmation) {
        $message = 'The password confirmation does not match.';
    } elseif (($passwordError = password_error($password)) !== null) {
        $message = $passwordError;
    } else {
        try {
            $conn->begin_transaction();
            $tokenHash = ResetToken::hash($token);
            $find = $conn->prepare('SELECT id_member FROM login WHERE reset_token = ? AND reset_token_expiry > UTC_TIMESTAMP() FOR UPDATE');
            $find->bind_param('s', $tokenHash);
            $find->execute();
            $user = $find->get_result()->fetch_assoc();
            if ($user === null) {
                $conn->rollback();
                $message = 'This reset link is invalid or has expired.';
            } else {
                $hash = password_hash($password, PASSWORD_DEFAULT);
                $update = $conn->prepare('UPDATE login SET password = ?, reset_token = NULL, reset_token_expiry = NULL WHERE id_member = ? AND reset_token = ?');
                $update->bind_param('sis', $hash, $user['id_member'], $tokenHash);
                $update->execute();
                if ($update->affected_rows !== 1) {
                    throw new RuntimeException('Reset token could not be consumed.');
                }
                $conn->commit();
                session_regenerate_id(true);
                $message = 'Your password has been reset. You may now sign in.';
                $token = '';
            }
        } catch (Throwable $exception) {
            try { $conn->rollback(); } catch (Throwable $ignored) { }
            error_log('Password reset failed: ' . $exception->getCode());
            $message = 'Unable to reset the password right now. Please request a new link.';
        }
    }
} else {
    $token = (string) ($_GET['token'] ?? '');
    if (!ResetToken::isWellFormed($token)) {
        $message = 'This reset link is invalid or has expired.';
        $token = '';
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset password</title>
    <link rel="stylesheet" href="assets/css/app.css">
    <script src="assets/js/password-toggle.js" defer></script>
</head>
<body>
<main class="container">
    <h1>Reset password</h1>
    <?php if ($message !== null): ?><p class="alert" role="status"><?= e($message) ?></p><?php endif; ?>
    <?php if ($token !== ''): ?>
    <form method="post" novalidate>
        <input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>">
        <input type="hidden" name="token" value="<?= e($token) ?>">
        <div class="form-group">
            <label for="password">New password</label>
            <div class="password-field">
                <input type="password" id="password" name="password" minlength="12" maxlength="4096" autocomplete="new-password" required>
                <button class="toggle-password" type="button" data-password-toggle="password" aria-label="Show password" aria-pressed="false">Show</button>
            </div>
        </div>
        <div class="form-group">
            <label for="password_confirmation">Confirm new password</label>
            <div class="password-field">
                <input type="password" id="password_confirmation" name="password_confirmation" minlength="12" maxlength="4096" autocomplete="new-password" required>
                <button class="toggle-password" type="button" data-password-toggle="password_confirmation" aria-label="Show password confirmation" aria-pressed="false">Show</button>
            </div>
        </div>
        <p class="hint">Use at least 12 characters.</p>
        <button type="submit">Reset password</button>
    </form>
    <?php endif; ?>
    <p><a href="login.php">Back to sign in</a></p>
</main>
</body>
</html>
