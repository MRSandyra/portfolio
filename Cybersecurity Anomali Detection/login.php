<?php
declare(strict_types=1);
require_once __DIR__ . '/security.php'; require_once __DIR__ . '/koneksi.php';
start_secure_session(); security_headers(); $message = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf()) { $message = 'Invalid request. Please try again.'; }
    else {
        $email = filter_var(trim((string) ($_POST['email'] ?? '')), FILTER_VALIDATE_EMAIL); $password = (string) ($_POST['password'] ?? '');
        if ($email === false || $password === '') { $message = 'Invalid email or password.'; }
        else {
            $stmt = $conn->prepare('SELECT id_member, full_name, password FROM login WHERE email = ? LIMIT 1'); $stmt->bind_param('s', $email); $stmt->execute(); $user = $stmt->get_result()->fetch_assoc();
            if ($user !== null && password_verify($password, $user['password'])) {
                session_regenerate_id(true); $_SESSION['user_id'] = (int) $user['id_member']; $_SESSION['full_name'] = (string) $user['full_name'];
                header('Location: index.php', true, 303); exit;
            }
            $message = 'Invalid email or password.';
        }
    }
}
?>
<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sign in</title><link rel="stylesheet" href="assets/css/app.css"></head>
<body><main class="container"><h1>Sign in</h1><?php if ($message): ?><p class="alert" role="alert"><?= e($message) ?></p><?php endif; ?>
<form method="post"><input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>"><div class="form-group"><label for="email">Email</label><input type="email" id="email" name="email" autocomplete="email" required></div><div class="form-group"><label for="password">Password</label><input type="password" id="password" name="password" autocomplete="current-password" required></div><button type="submit">Sign in</button></form>
<p><a href="Reset_form.php">Forgot password?</a> · <a href="register.php">Create account</a></p></main></body></html>
