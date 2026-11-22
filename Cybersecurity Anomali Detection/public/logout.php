<?php
require_once dirname(__DIR__) . '/src/Security.php';
start_secure_session();
$_SESSION = [];
session_unset();
session_destroy();

header('Location: index.php', true, 303);
exit();
