<?php
declare(strict_types=1);
require_once __DIR__ . '/security.php'; require_once __DIR__ . '/AnalysisWorkspace.php';
start_secure_session(); security_headers();
$job = (string) ($_GET['job'] ?? ''); $createdAt = $_SESSION['analysis_jobs'][$job] ?? null;
$workspace = is_int($createdAt) && $createdAt >= time() - 3600 ? AnalysisWorkspace::pathFor($job) : null;
$file = $workspace === null ? null : $workspace . DIRECTORY_SEPARATOR . 'attack_detection_results.csv';
if ($file === null || !is_file($file)) { http_response_code(404); exit('Result not found or expired.'); }
header('Content-Type: text/csv; charset=UTF-8'); header('Content-Disposition: attachment; filename="attack_detection_results.csv"'); header('Content-Length: ' . (string) filesize($file));
readfile($file); exit;
