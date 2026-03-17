<?php if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $token = $_POST['token'];
    $newPassword = password_hash($_POST['new_password'], PASSWORD_DEFAULT);
    
    // Cek token di database
    // Jika valid, update password
    // Hapus token dari database
    
    echo "Password berhasil diperbarui.";
}
?>