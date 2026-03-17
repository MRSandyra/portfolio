<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Password</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <link rel="stylesheet" href="assets/css/reset_password.css">
</head>
<body>
    <div class="container">
        <h2>Reset Password</h2>

        <?php
        include 'koneksi.php'; // Pastikan koneksi database ($conn) sudah diinisialisasi

        if ($_SERVER['REQUEST_METHOD'] === 'POST') {
            // Proses pengiriman form reset password
            $token = $_POST['token'];
            $password = $_POST['password'];
            $confirmPassword = $_POST['confirm-password'];

            // Validasi password dan konfirmasi password
            if ($password !== $confirmPassword) {
                echo "<p>Password tidak cocok</p>";
                exit;
            }

            // Debugging: Cetak token yang diterima
            echo "Token diterima: " . htmlspecialchars($token) . "<br>";

            // Cek token dan waktu kedaluwarsa
            $stmt = $conn->prepare("SELECT id_member, reset_token_expiry FROM login WHERE reset_token = ? AND reset_token_expiry > NOW()");
            $stmt->bind_param("s", $token);
            $stmt->execute();
            $result = $stmt->get_result();
            $user = $result->fetch_assoc();

            if ($user) {
                // Debugging: Token valid
                echo "Token valid, id_member: " . $user['id_member'] . "<br>";
                echo "Reset Token Expiry: " . $user['reset_token_expiry'] . "<br>";
                echo "Current Time: " . date('Y-m-d H:i:s') . "<br>";

                // Proses reset password
                $hashedPassword = password_hash($password, PASSWORD_BCRYPT);

                // Update password dan hapus token dari database
                $stmt = $conn->prepare("UPDATE login SET password = ?, reset_token = NULL, reset_token_expiry = NULL WHERE id_member = ?");
                $stmt->bind_param("si", $hashedPassword, $user['id_member']);
                $stmt->execute();

                echo "<p>Password berhasil direset.</p>";
            } else {
                // Debugging: Token tidak valid atau sudah kadaluarsa
                echo "<p>Token tidak valid atau sudah kadaluarsa.</p>";
                echo "Query Error: " . $conn->error . "<br>";
                echo "Current Time: " . date('Y-m-d H:i:s') . "<br>";
            }
        } else {
            // Jika method GET, tampilkan form reset password
            if (isset($_GET['token'])) {
                $token = $_GET['token'];
                ?>

                <form action="" method="post">
                    <input type="hidden" name="token" value="<?php echo htmlspecialchars($token); ?>">

                    <div class="form-group">
                        <label for="password">Password *</label>
                        <div class="input-container">
                            <input type="password" id="password" name="password" required>
                            <span class="toggle-password" onclick="togglePassword('password', this)">
                                <i class="fas fa-eye"></i>
                            </span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="confirm-password">Confirm Password *</label>
                        <div class="input-container">
                            <input type="password" id="confirm-password" name="confirm-password" required>
                            <span class="toggle-password" onclick="togglePassword('confirm-password', this)">
                                <i class="fas fa-eye"></i>
                            </span>
                        </div>
                    </div>

                    <button type="submit" class="btn">Reset Password</button>
                </form>

                <?php
            } else {
                echo "<p>Token tidak ditemukan. Silakan periksa kembali link yang diberikan.</p>";
            }
        }
        ?>

    </div>

    <script>
        function togglePassword(fieldId, toggleIcon) {
            var field = document.getElementById(fieldId);
            var icon = toggleIcon.querySelector("i");
            if (field.type === "password") {
                field.type = "text";
                icon.classList.remove("fa-eye");
                icon.classList.add("fa-eye-slash");
            } else {
                field.type = "password";
                icon.classList.remove("fa-eye-slash");
                icon.classList.add("fa-eye");
            }
        }
    </script>
</body>
</html>
