<?php
declare(strict_types=1);

require_once __DIR__ . '/Security.php';

final class ResetToken
{
    /** @return array{token:string,hash:string,expiresAt:string} */
    public static function issue(?DateTimeImmutable $now = null): array
    {
        $now ??= new DateTimeImmutable('now', new DateTimeZone('UTC'));
        $token = rtrim(strtr(base64_encode(random_bytes(32)), '+/', '-_'), '=');
        return [
            'token' => $token,
            'hash' => self::hash($token),
            'expiresAt' => $now->modify('+' . RESET_TOKEN_TTL_SECONDS . ' seconds')->format('Y-m-d H:i:s'),
        ];
    }

    public static function hash(string $token): string
    {
        return hash('sha256', $token);
    }

    public static function isWellFormed(string $token): bool
    {
        return (bool) preg_match('/^[A-Za-z0-9_-]{43}$/D', $token);
    }
}
