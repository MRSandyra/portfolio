import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE = '127.0.0.1 - - [01/Jan/2026:00:00:00 +0000] "GET /?q=SELECT HTTP/1.1" 200 1 "-" "test"\n'


class AnalyzerIsolationTest(unittest.TestCase):
    def test_two_requests_keep_separate_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_one, input_two = root / 'one.log', root / 'two.log'
            input_one.write_text(LINE, encoding='utf-8')
            input_two.write_text(LINE.replace('SELECT', 'normal'), encoding='utf-8')
            out_one, out_two = root / 'job-one', root / 'job-two'
            commands = [
                [sys.executable, str(ROOT / 'check.py'), str(input_one), '--output-dir', str(out_one)],
                [sys.executable, str(ROOT / 'check.py'), str(input_two), '--output-dir', str(out_two)],
            ]
            processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for command in commands]
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, (stdout + stderr).decode())
            self.assertTrue((out_one / 'attack_detection_results.csv').is_file())
            self.assertTrue((out_two / 'attack_detection_results.csv').is_file())
            self.assertNotEqual((out_one / 'result.txt').read_text(), (out_two / 'result.txt').read_text())
            self.assertFalse((root / 'result.txt').exists())


if __name__ == '__main__':
    unittest.main()
