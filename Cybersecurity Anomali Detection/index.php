<?php
session_start(); // Memulai session

// Fungsi untuk mengambil huruf awal dari full_name
function get_initial($full_name) {
    $words = explode(' ', $full_name); // Pisahkan nama berdasarkan spasi
    $initials = '';
    foreach ($words as $word) {
        $initials .= strtoupper($word[0]); // Ambil huruf pertama dari setiap kata dan buat huruf besar
    }
    return $initials[0]; // Ambil huruf pertama dari initial
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anomaly Checker</title>
    <link rel="stylesheet" href="assets/css/index.css">
</head>
<body>
    <div class="navbar">
        <h1>Anomaly Checker</h1>
        <?php if(isset($_SESSION['full_name'])): ?>
            <div class="dropdown">
                <button class="dropbtn">
                    <?php echo get_initial($_SESSION['full_name']); ?>
                </button>
                <div class="dropdown-content">
                    <a href="account_settings.php">Account Settings</a>
                    <a href="logout.php">Logout</a>
                </div>
            </div>
        <?php else: ?>
            <a href="login.php">Login</a>
        <?php endif; ?>
    </div>

    <div class="container">
        <div class="upload-box">
            <h1>Upload access.log File</h1>
            <form action="result.php" method="POST" enctype="multipart/form-data">
                <label for="access_log">Choose a file</label>
                <input type="file" name="logfile" id="logfile" accept=".log" required>
                <br>
                <button type="submit">Upload and Check Anomalies</button>
            </form>
        </div>

        <div class="info-toggle">
            <button id="toggle-info">Learn About Detected Attacks</button>
        </div>

        <div class="info-box" id="info-box">
            <h2>Types of Attacks Detected</h2>
            <div class="attack-info">
                <div class="attack-item">
                    <h3>SQL Injection</h3>
                    <p>Identifies malicious SQL statements that manipulate databases.</p>
                </div>
                <div class="attack-item">
                    <h3>Path Traversal</h3>
                    <p>Detects attempts to access restricted directories and files.</p>
                </div>
                <div class="attack-item">
                    <h3>XSS (Cross-Site Scripting)</h3>
                    <p>Protects against scripts injected into web pages to compromise users.</p>
                </div>
                <div class="attack-item">
                    <h3>Remote File Inclusion (RFI)</h3>
                    <p>Monitors attempts to include external files on the server.</p>
                </div>
                <div class="attack-item">
                    <h3>Malicious Payloads</h3>
                    <p>Flags potential shell commands or code injections in requests.</p>
                </div>
                <div class="attack-item">
                    <h3>HTTP Methods Abuse</h3>
                    <p>Detects misuse of HTTP methods like TRACE, TRACK, and more.</p>
                </div>
                <div class="attack-item">
                    <h3>Password-Based Attacks</h3>
                    <p>Identifies brute force or repeated login attempts.</p>
                </div>
                <div class="attack-item">
                    <h3>Denial of Service (DoS)</h3>
                    <p>Recognizes flooding attacks aimed at overwhelming the server.</p>
                </div>
            </div>
        </div>
    </div>

    <footer>
        &copy; 2024 Anomaly Checker. All rights reserved.
    </footer>

    <script>
        document.getElementById('toggle-info').addEventListener('click', function() {
            var infoBox = document.getElementById('info-box');
            if (infoBox.style.display === 'none' || infoBox.style.display === '') {
                infoBox.style.display = 'block';
            } else {
                infoBox.style.display = 'none';
            }
        });
    </script>
    <script src="assets/js/script.js"></script>
</body>
</html>
