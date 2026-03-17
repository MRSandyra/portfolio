<?php
include 'koneksi.php'; // Menyertakan file koneksi

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $full_name = $_POST['full_name'];
    $email = $_POST['email'];
    $password = $_POST['password'];

    // Hash password
    $hashed_password = password_hash($password, PASSWORD_DEFAULT);

    // Menghindari SQL Injection
    $full_name = $conn->real_escape_string($full_name);
    $email = $conn->real_escape_string($email);
    $hashed_password = $conn->real_escape_string($hashed_password);

    // SQL query untuk memasukkan data ke tabel login
    $sql = "INSERT INTO login (full_name, email, password) VALUES ('$full_name', '$email', '$hashed_password')";

    // Eksekusi query
    if ($conn->query($sql) === TRUE) {
        echo "Akun berhasil dibuat. Silakan <a href='login.php'>login</a>.";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }

    $conn->close();
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Create Account</title>
    <link rel="stylesheet" type="text/css" href="assets/css/Register.css"> 
</head>
<body>
    <div class="container">
        <h1>Create an Account</h1>
        <form method="post" action="register.php">
            <div class="form-group">
                <label for="full_name">Full Name</label>
                <input type="text" id="full_name" name="full_name" required>
            </div>
            <div class="form-group">
                <label for="email">Email</label>
                <input type="email" id="email" name="email" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">CREATE ACCOUNT</button>
        </form>
    </div>
</body>
</html>
