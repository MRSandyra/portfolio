<?php
include 'koneksi.php'; // Menyertakan file koneksi

// Mulai sesi
session_start();

// Mengecek apakah pengguna sudah login
if (!isset($_SESSION['full_name'])) {
    header("Location: login.php");
    exit();
}

$full_name = $_SESSION['full_name'];

// Mengambil data pengguna dari database
if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    $sql = "SELECT email FROM login WHERE full_name = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param('s', $full_name);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        $user = $result->fetch_assoc();
        $email = $user['email'];
    } else {
        echo "User not found.";
    }
    $stmt->close();
}

// Memproses form jika di-submit
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $new_full_name = $_POST['full_name'];
    $new_email = $_POST['email'];
    $current_password = $_POST['current_password'];
    $new_password = $_POST['new_password'];
    $confirm_password = $_POST['confirm_password'];

    // Memverifikasi password dan memperbarui data
    $sql = "SELECT password FROM login WHERE full_name = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param('s', $full_name);
    $stmt->execute();
    $result = $stmt->get_result();
    $user = $result->fetch_assoc();
    
    if (password_verify($current_password, $user['password'])) {
        // Update data pengguna
        if (!empty($new_password) && $new_password === $confirm_password) {
            $new_password_hash = password_hash($new_password, PASSWORD_BCRYPT);
            $sql = "UPDATE login SET full_name = ?, email = ?, password = ? WHERE full_name = ?";
            $stmt = $conn->prepare($sql);
            $stmt->bind_param('ssss', $new_full_name, $new_email, $new_password_hash, $full_name);
        } else {
            $sql = "UPDATE login SET full_name = ?, email = ? WHERE full_name = ?";
            $stmt->prepare($sql);
            $stmt->bind_param('sss', $new_full_name, $new_email, $full_name);
        }
        $stmt->execute();
        $stmt->close();
        echo "Settings updated successfully.";
        
        // Update session variable
        $_SESSION['full_name'] = $new_full_name;
    } else {
        echo "Current password is incorrect.";
    }
}

$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Account Settings</title>
    <link rel="stylesheet" href="assets/css/account.css">
</head>
<body>

    <div class="settings-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <!-- Tombol Back -->
            <div class="back-button">
                <a href="index.php" class="back-link">← Back to Dashboard</a>
            </div>

            <ul>
                <li><a href="#profile-info">Profile Information</a></li>
                <li><a href="#password-change">Change Password</a></li>
                <li><a href="#notifications">Notifications</a></li>
            </ul>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <div class="settings-header">
                <h1>Account Settings</h1>
            </div>

            <!-- Profile Information Section -->
            <div class="settings-section" id="profile-info">
                <h2>Profile Information</h2>
                <form action="account_settings.php" method="POST">
                    <label for="full_name">Full Name</label>
                    <input type="text" id="full_name" name="full_name" value="<?php echo htmlspecialchars($full_name); ?>">

                    <label for="email">Email Address</label>
                    <input type="email" id="email" name="email" value="<?php echo htmlspecialchars($email); ?>">

                    <button type="submit">Save Changes</button>
                </form>
            </div>

            <!-- Change Password Section -->
            <div class="settings-section" id="password-change">
                <h2>Change Password</h2>
                <form action="account_settings.php" method="POST">
                    <label for="current_password">Current Password</label>
                    <input type="password" id="current_password" name="current_password">

                    <label for="new_password">New Password</label>
                    <input type="password" id="new_password" name="new_password">

                    <label for="confirm_password">Confirm New Password</label>
                    <input type="password" id="confirm_password" name="confirm_password">

                    <button type="submit">Update Password</button>
                </form>
            </div>

            <!-- Notifications Section -->
            <div class="settings-section" id="notifications">
                <h2>Notifications</h2>
                <label for="email_notifications">Email Notifications</label>
                <input type="checkbox" id="email_notifications" name="email_notifications"> Enable email notifications
            </div>
        </div>
    </div>

</body>
</html>
