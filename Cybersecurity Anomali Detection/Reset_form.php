<?php
require_once __DIR__ . '/security.php';
start_secure_session();
security_headers();
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Request password reset</title>
  <link rel="stylesheet" href="assets/css/app.css">
</head>
<body>
  <div class="container">
    <h1>Reset Password</h1>
    <form action="send_reset_link.php" method="post">
      <input type="hidden" name="csrf_token" value="<?= e(csrf_token()) ?>">
      <div class="form-group">
        <label for="email">E-mail</label>
        <input type="email" id="email" name="email" required>
      </div>

      <button type="submit">Send reset link</button>
    </form>

    <?php if (isset($_SESSION['reset_request_message'])): ?>
      <div class="alert">
        <?= e((string) $_SESSION['reset_request_message']) ?>
      </div>
      <?php unset($_SESSION['reset_request_message']); ?>
    <?php endif; ?>
  </div>
</body>
</html>

