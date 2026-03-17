<!DOCTYPE html>
<html>
<head>
  <title>Reset Password</title>
  <link rel="stylesheet" type="text/css" href="assets/CSS/Register.css"> 
</head>
<body>
  <div class="container">
    <h1>Reset Password</h1>
    <form action="reset2.php" method="post">
      <div class="form-group">
        <label for="email">E-mail</label>
        <input type="email" id="email" name="email" required>
      </div>

      <button type="submit">Reset Password</button>
    </form>

    <!-- Area untuk menampilkan pesan error atau sukses -->
    <?php if (isset($_GET['message'])): ?>
      <div class="alert">
        <?php echo htmlspecialchars($_GET['message']); ?>
      </div>
    <?php endif; ?>
  </div>
</body>
</html>

