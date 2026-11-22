<?php
declare(strict_types=1);
require_once dirname(__DIR__) . '/src/Security.php'; require_once dirname(__DIR__) . '/config/database.php';
start_secure_session(); security_headers();
if (!isset($_SESSION['user_id'])) { header('Location: login.php', true, 303); exit; }
$userId = (int) $_SESSION['user_id']; $message = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf()) { $message = 'Invalid request. Please try again.'; }
    elseif (($_POST['action'] ?? '') === 'profile') {
        $name = trim((string) ($_POST['full_name'] ?? '')); $email = filter_var(trim((string) ($_POST['email'] ?? '')), FILTER_VALIDATE_EMAIL);
        if ($name === '' || strlen($name) > 255 || $email === false) { $message = 'Enter a valid name and email address.'; }
        else { try { $stmt = $conn->prepare('UPDATE login SET full_name = ?, email = ? WHERE id_member = ?'); $stmt->bind_param('ssi', $name, $email, $userId); $stmt->execute(); $_SESSION['full_name'] = $name; $message = 'Profile updated.'; } catch (mysqli_sql_exception $exception) { error_log('Profile update failed: ' . $exception->getCode()); $message = 'Unable to update the profile.'; } }
    } elseif (($_POST['action'] ?? '') === 'password') {
        $current = (string) ($_POST['current_password'] ?? ''); $new = (string) ($_POST['new_password'] ?? ''); $confirm = (string) ($_POST['confirm_password'] ?? '');
        $stmt = $conn->prepare('SELECT password FROM login WHERE id_member = ?'); $stmt->bind_param('i', $userId); $stmt->execute(); $user = $stmt->get_result()->fetch_assoc();
        if ($user === null || !password_verify($current, $user['password'])) { $message = 'Current password is incorrect.'; }
        elseif ($new !== $confirm) { $message = 'The password confirmation does not match.'; }
        elseif (($passwordError = password_error($new)) !== null) { $message = $passwordError; }
        else { $hash = password_hash($new, PASSWORD_DEFAULT); $update = $conn->prepare('UPDATE login SET password = ? WHERE id_member = ?'); $update->bind_param('si', $hash, $userId); $update->execute(); $message = 'Password updated.'; }
    }
}
$stmt = $conn->prepare('SELECT full_name, email FROM login WHERE id_member = ?'); $stmt->bind_param('i', $userId); $stmt->execute(); $profile = $stmt->get_result()->fetch_assoc();
if ($profile === null) { session_destroy(); header('Location: login.php', true, 303); exit; }
?>
<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Account settings</title><link rel="stylesheet" href="assets/css/app.css"></head>
<body><main class="container"><h1>Account settings</h1><?php if ($message): ?><p class="alert" role="status"><?= e($message) ?></p><?php endif; ?>
<h2>Profile</h2><form method="post"><input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>"><input type="hidden" name="action" value="profile"><div class="form-group"><label for="full_name">Full name</label><input id="full_name" name="full_name" maxlength="255" value="<?= e($profile['full_name']) ?>" required></div><div class="form-group"><label for="email">Email</label><input type="email" id="email" name="email" value="<?= e($profile['email']) ?>" required></div><button>Save profile</button></form>
<h2>Change password</h2><form method="post"><input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>"><input type="hidden" name="action" value="password"><div class="form-group"><label for="current_password">Current password</label><input type="password" id="current_password" name="current_password" autocomplete="current-password" required></div><div class="form-group"><label for="new_password">New password</label><input type="password" id="new_password" name="new_password" minlength="12" maxlength="4096" autocomplete="new-password" required></div><div class="form-group"><label for="confirm_password">Confirm password</label><input type="password" id="confirm_password" name="confirm_password" minlength="12" maxlength="4096" autocomplete="new-password" required></div><button>Update password</button></form><p><a href="index.php">Back to dashboard</a></p></main></body></html>
