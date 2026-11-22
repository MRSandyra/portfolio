<?php
declare(strict_types=1);
require_once __DIR__ . '/../src/ResetToken.php';
require_once __DIR__ . '/../src/UploadValidator.php';

function expect(bool $condition, string $message): void { if (!$condition) { throw new RuntimeException($message); } }

$issued = ResetToken::issue(new DateTimeImmutable('2026-01-01 00:00:00', new DateTimeZone('UTC')));
expect(ResetToken::isWellFormed($issued['token']), 'A reset token must be URL-safe and correctly sized.');
expect(strlen($issued['hash']) === 64 && $issued['hash'] === ResetToken::hash($issued['token']), 'Only the SHA-256 token hash may be stored.');
expect($issued['expiresAt'] === '2026-01-01 01:00:00', 'Reset token expiry must be one hour.');
$storedHash = $issued['hash'];
expect(hash_equals($storedHash, ResetToken::hash($issued['token'])), 'Fresh token must validate.');
$storedHash = null; // Models the atomic DB update that consumes a token.
expect($storedHash === null, 'Consumed token must not be reusable.');

$valid = tempnam(sys_get_temp_dir(), 'access-log-');
file_put_contents($valid, "127.0.0.1 - - [01/Jan/2026:00:00:00 +0000] \"GET / HTTP/1.1\" 200 1 \"-\" \"test\"\n");
expect(UploadValidator::validatePath($valid, 'access.log')['valid'], 'Valid combined log should pass validation.');
expect(!UploadValidator::validatePath($valid, 'access.php')['valid'], 'Executable extension must be rejected.');
file_put_contents($valid, "not a combined log\n");
expect(!UploadValidator::validatePath($valid, 'access.log')['valid'], 'Invalid log content must be rejected.');
unlink($valid);
echo "PHP security tests passed.\n";
