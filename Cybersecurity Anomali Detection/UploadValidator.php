<?php
declare(strict_types=1);

require_once __DIR__ . '/security.php';

final class UploadValidator
{
    /** @return array{valid:bool,error:?string} */
    public static function validate(array $upload): array
    {
        if (!isset($upload['error'], $upload['tmp_name'], $upload['name'], $upload['size']) || $upload['error'] !== UPLOAD_ERR_OK) {
            return ['valid' => false, 'error' => 'Upload failed. Please choose a valid log file.'];
        }
        if (!is_int($upload['size']) && !ctype_digit((string) $upload['size'])) {
            return ['valid' => false, 'error' => 'Invalid upload metadata.'];
        }
        if ((int) $upload['size'] < 1 || (int) $upload['size'] > MAX_UPLOAD_BYTES) {
            return ['valid' => false, 'error' => 'Log files must be between 1 byte and 5 MB.'];
        }
        return self::validatePath((string) $upload['tmp_name'], (string) $upload['name']);
    }

    /** @return array{valid:bool,error:?string} */
    public static function validatePath(string $path, string $originalName): array
    {
        $extension = strtolower(pathinfo($originalName, PATHINFO_EXTENSION));
        if (!in_array($extension, ['log', 'txt'], true)) {
            return ['valid' => false, 'error' => 'Only .log and .txt access-log files are accepted.'];
        }
        if (!is_file($path) || !is_readable($path)) {
            return ['valid' => false, 'error' => 'Uploaded file could not be read.'];
        }
        $size = filesize($path);
        if ($size === false || $size < 1 || $size > MAX_UPLOAD_BYTES) {
            return ['valid' => false, 'error' => 'Log files must be between 1 byte and 5 MB.'];
        }
        $sample = file_get_contents($path, false, null, 0, 16384);
        if ($sample === false || str_contains($sample, "\0")) {
            return ['valid' => false, 'error' => 'The file must be plain-text access-log data.'];
        }
        $mime = (new finfo(FILEINFO_MIME_TYPE))->file($path);
        if (!in_array($mime, ['text/plain', 'text/x-log', 'application/octet-stream'], true)) {
            return ['valid' => false, 'error' => 'The file MIME type is not a supported text log.'];
        }
        $combinedLog = '/^\S+\s+\S+\s+\S+\s+\[[^\]]+\]\s+"[A-Z]+\s+[^\s]+\s+HTTP\/\d(?:\.\d)?"\s+\d{3}\s+(?:\d+|-)\s+"[^"]*"\s+"[^"]*"/m';
        if (!preg_match($combinedLog, $sample)) {
            return ['valid' => false, 'error' => 'The file does not appear to use Apache/Nginx combined log format.'];
        }
        return ['valid' => true, 'error' => null];
    }
}
