<? if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $email = $_POST['email'];

    // Validate email address
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        echo "Invalid email address";
        exit;
    }

    // Check if email exists in database using a prepared statement
    $stmt = $pdo->prepare("SELECT 1 FROM login WHERE email = :email");
    $stmt->bindParam(":email", $email);
    $stmt->execute();
    if (!$stmt->fetch()) {
        echo "Email not found";
        exit;
    }

    // Generate a secure token
    $token = password_hash(random_bytes(50), PASSWORD_BCRYPT);

    // Update database with token and timestamp
    $stmt = $pdo->prepare("UPDATE login SET reset_token = :token, reset_timestamp = :timestamp WHERE email = :email");
    $stmt->bindParam(":token", $token);
    $stmt->bindParam(":timestamp", time());
    $stmt->bindParam(":email", $email);
    $stmt->execute();

    // Send email with reset link
    $resetLink = "https://example.com/reset_password.php?token=" . $token;
    $subject = "Reset Password";
    $message = "Klik link ini untuk mereset password Anda: " . $resetLink;
    if (!mail($email, $subject, $message)) {
        echo "Error sending email";
        exit;
    }

    echo "Link reset password telah dikirim ke email Anda.";
}