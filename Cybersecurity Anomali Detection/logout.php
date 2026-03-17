<?php
session_start();
session_unset();
session_destroy();

// Redirect ke halaman index.php
header("Location: index.php");
exit();
?>
