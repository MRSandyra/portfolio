<?php
declare(strict_types=1);

final class AnalysisWorkspace
{
    private const TTL_SECONDS = 3600;
    public static function root(): string
    {
        $configured = (string) getenv('ANALYSIS_WORKSPACE_ROOT');
        $root = $configured !== '' ? $configured : sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'cybersecurity-anomaly-analysis';
        if (!is_dir($root) && !mkdir($root, 0700, true) && !is_dir($root)) { throw new RuntimeException('Cannot create analysis workspace root.'); }
        return realpath($root) ?: $root;
    }
    public static function create(): array
    {
        self::cleanup(); $root = self::root();
        for ($attempt = 0; $attempt < 3; $attempt++) {
            $id = bin2hex(random_bytes(16)); $path = $root . DIRECTORY_SEPARATOR . $id;
            if (mkdir($path, 0700)) { return ['id' => $id, 'path' => $path]; }
        }
        throw new RuntimeException('Cannot create analysis workspace.');
    }
    public static function pathFor(string $id): ?string
    {
        if (!preg_match('/^[a-f0-9]{32}$/D', $id)) { return null; }
        $path = self::root() . DIRECTORY_SEPARATOR . $id;
        return is_dir($path) ? $path : null;
    }
    public static function cleanup(): void
    {
        $root = self::root();
        foreach (new DirectoryIterator($root) as $entry) {
            if ($entry->isDot() || !$entry->isDir() || !preg_match('/^[a-f0-9]{32}$/D', $entry->getFilename())) { continue; }
            if ($entry->getMTime() < time() - self::TTL_SECONDS) { self::removeDirectory($entry->getPathname(), $root); }
        }
    }
    private static function removeDirectory(string $path, string $root): void
    {
        $realPath = realpath($path);
        if ($realPath === false || dirname($realPath) !== $root) { return; }
        $iterator = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($realPath, FilesystemIterator::SKIP_DOTS), RecursiveIteratorIterator::CHILD_FIRST);
        foreach ($iterator as $item) { $item->isDir() ? rmdir($item->getPathname()) : unlink($item->getPathname()); }
        rmdir($realPath);
    }
}
