<?php
$servername = "localhost";
$username = "root";      // Ubah sesuai dengan username database MySQL kamu
$password = "";          // Ubah sesuai dengan password database MySQL kamu
$database = "anomali_detection";

// Membuat koneksi
$conn = new mysqli($servername, $username, $password, $database);

// Mengecek koneksi
if ($conn->connect_error) {
    die("Koneksi gagal: " . $conn->connect_error);
}
?>
