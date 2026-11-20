<?php
require_once __DIR__ . '/security.php';
start_secure_session();
$_SESSION = [];
session_unset();
session_destroy();

header('Location: index.php', true, 303);
exit();
