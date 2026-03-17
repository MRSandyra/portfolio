<?php
include 'koneksi.php';

$response = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $email = $_POST['email'];

    // Generate unique token
    $reset_token = bin2hex(random_bytes(16)); // Generate a 32-character token
    $expiry_time = date('Y-m-d H:i:s', strtotime('+1 hour')); // Token valid for 1 hour

    // Store the token and expiry time in the database
    $stmt = $conn->prepare("UPDATE login SET reset_token = ?, reset_token_expiry = ? WHERE email = ?");
    $stmt->bind_param("sss", $reset_token, $expiry_time, $email);
    if ($stmt->execute()) {
        // Run send-email.js with Node.js to send email with the token
        $command = 'node C:/xampp/htdocs/anomali_detection/assets/js/send-email.js ' . escapeshellarg($email) . ' ' . escapeshellarg($reset_token);
        exec($command, $output, $resultCode);

        if ($resultCode === 0) {
            $response = [
                "success" => 1,
                "message" => "Email reset password telah dikirim."
            ];
        } else {
            $response = [
                "success" => 0,
                "message" => "Maaf, terjadi kesalahan sistem. Permintaan Anda tidak dapat diproses."
            ];
        }
    } else {
        $response = [
            "success" => 0,
            "message" => "Maaf, terjadi kesalahan sistem. Permintaan Anda tidak dapat diproses."
        ];
    }
}

// Encode response as JSON and pass it to JavaScript
echo '<script type="text/javascript">',
     'var response = ' . json_encode($response) . ';',
     '</script>';
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Password</title>
    <script>
        // Function to show alert popup based on PHP response
        function showPopup() {
            if (response) {
                alert(response.message);
                if (response.success === 1) {
                    // Redirect to Gmail after clicking OK
                    window.location.href = "https://mail.google.com/";
                }
            }
        }
    </script>
</head>
<body onload="showPopup()">
    <div class="container">
        <h2>Reset Password</h2>
        <form action="reset2.php" method="post">
            <div class="form-group">
                <label for="email">E-mail</label>
                <input type="email" id="email" name="email" required>
            </div>
            <button type="submit" class="btn">Reset Password</button>
        </form>
    </div>
</body>
</html>
