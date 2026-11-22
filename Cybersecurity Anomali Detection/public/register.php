<?php
declare(strict_types=1);
require_once dirname(__DIR__) . '/src/Security.php'; require_once dirname(__DIR__) . '/config/database.php';
start_secure_session(); security_headers(); $message = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!verify_csrf()) { $message = 'Invalid request. Please try again.'; }
    else {
        $name = trim((string) ($_POST['full_name'] ?? '')); $email = filter_var(trim((string) ($_POST['email'] ?? '')), FILTER_VALIDATE_EMAIL); $password = (string) ($_POST['password'] ?? '');
        if ($name === '' || strlen($name) > 255 || $email === false) { $message = 'Enter a valid name and email address.'; }
        elseif (($passwordError = password_error($password)) !== null) { $message = $passwordError; }
        else {
            try { $hash = password_hash($password, PASSWORD_DEFAULT); $stmt = $conn->prepare('INSERT INTO login (full_name, email, password) VALUES (?, ?, ?)'); $stmt->bind_param('sss', $name, $email, $hash); $stmt->execute(); header('Location: login.php', true, 303); exit; }
            catch (mysqli_sql_exception $exception) { error_log('Registration failed: ' . $exception->getCode()); $message = 'Unable to create the account with those details.'; }
        }
    }
}
?>
<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Create account</title><link rel="stylesheet" href="assets/css/app.css"></head>
<body><main class="container"><h1>Create an account</h1><?php if ($message): ?><p class="alert" role="alert"><?= e($message) ?></p><?php endif; ?>
<form method="post"><input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>"><div class="form-group"><label for="full_name">Full name</label><input type="text" id="full_name" name="full_name" maxlength="255" autocomplete="name" required></div><div class="form-group"><label for="email">Email</label><input type="email" id="email" name="email" autocomplete="email" required></div><div class="form-group"><label for="password">Password</label><input type="password" id="password" name="password" minlength="12" maxlength="4096" autocomplete="new-password" required></div><button type="submit">Create account</button></form>
<p><a href="login.php">Back to sign in</a></p></main></body></html>
