<?php
// Legacy endpoint: password resets are handled atomically by reset_password.php.
header('Location: Reset_form.php', true, 303);
exit;
