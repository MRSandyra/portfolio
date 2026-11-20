<?php
// Legacy endpoint retained for bookmarked forms; all requests use the hardened handler.
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    require __DIR__ . '/send_reset_link.php';
    exit;
}
header('Location: Reset_form.php', true, 303);
exit;
