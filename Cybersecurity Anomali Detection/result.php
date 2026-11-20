<?php
declare(strict_types=1);
require_once __DIR__ . '/security.php';
require_once __DIR__ . '/UploadValidator.php';
require_once __DIR__ . '/AnalysisWorkspace.php';
start_secure_session(); security_headers();
$error = null; $rows = []; $jobId = null;
if ($_SERVER['REQUEST_METHOD'] !== 'POST' || !verify_csrf()) {
    $error = 'Invalid upload request. Please try again.';
} else {
    $validation = UploadValidator::validate($_FILES['logfile'] ?? []);
    if (!$validation['valid']) { $error = $validation['error']; }
    else {
        try {
            $workspace = AnalysisWorkspace::create();
            $input = $workspace['path'] . DIRECTORY_SEPARATOR . 'input.log';
            if (!move_uploaded_file((string) $_FILES['logfile']['tmp_name'], $input)) { throw new RuntimeException('Could not store the upload.'); }
            chmod($input, 0600);
            $python = (string) (getenv('PYTHON_BIN') ?: 'python3');
            $command = [$python, __DIR__ . DIRECTORY_SEPARATOR . 'check.py', $input, '--output-dir', $workspace['path']];
            $process = proc_open($command, [0 => ['pipe', 'r'], 1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes, null, null, ['bypass_shell' => true]);
            if (!is_resource($process)) { throw new RuntimeException('Analyzer could not start.'); }
            fclose($pipes[0]); stream_set_blocking($pipes[1], false); stream_set_blocking($pipes[2], false);
            $deadline = microtime(true) + max(5, min(120, (int) (getenv('ANALYSIS_TIMEOUT_SECONDS') ?: 30)));
            while (($status = proc_get_status($process))['running']) {
                if (microtime(true) >= $deadline) { proc_terminate($process, 9); throw new RuntimeException('Analyzer timed out.'); }
                usleep(100000);
            }
            $stdout = stream_get_contents($pipes[1]); $stderr = stream_get_contents($pipes[2]); fclose($pipes[1]); fclose($pipes[2]);
            $exitCode = $status['exitcode']; proc_close($process);
            if ($exitCode !== 0 || !is_file($workspace['path'] . DIRECTORY_SEPARATOR . 'result.txt')) {
                error_log('Analyzer failed: ' . substr((string) $stderr . (string) $stdout, 0, 1000)); throw new RuntimeException('Analyzer could not process this file.');
            }
            foreach (file($workspace['path'] . DIRECTORY_SEPARATOR . 'result.txt', FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: [] as $line) {
                [$label, $count] = array_pad(explode(':', $line, 2), 2, '');
                if (preg_match('/^Total [A-Za-z() -]+$/D', $label) && ctype_digit(trim($count))) { $rows[] = ['label' => $label, 'count' => trim($count)]; }
            }
            if ($rows === [] || !is_file($workspace['path'] . DIRECTORY_SEPARATOR . 'attack_detection_results.csv')) { throw new RuntimeException('Analyzer generated incomplete output.'); }
            $_SESSION['analysis_jobs'][$workspace['id']] = time(); $jobId = $workspace['id'];
        } catch (Throwable $exception) { error_log('Upload analysis failed: ' . $exception->getCode()); $error = 'We could not analyze that file. Confirm it is a supported access log and try again.'; }
    }
}
?>
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Analysis results</title><link rel="stylesheet" href="assets/css/app.css"></head>
<body><main class="container"><h1>Analysis results</h1>
<?php if ($error !== null): ?><p class="alert" role="alert"><?= e($error) ?></p><?php else: ?>
<p class="success" role="status">Your log was analyzed in an isolated workspace.</p><table><thead><tr><th scope="col">Attack type</th><th scope="col">Count</th></tr></thead><tbody>
<?php foreach ($rows as $row): ?><tr><td><?= e($row['label']) ?></td><td><?= e($row['count']) ?></td></tr><?php endforeach; ?>
</tbody></table><p><a href="download.php?job=<?= e((string) $jobId) ?>">Download detailed CSV</a></p>
<?php endif; ?><p><a href="index.php">Upload another file</a></p></main></body></html>
