import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "tools/data_process/pai_http_range.py"
SPEC = importlib.util.spec_from_file_location("pai_http_range", SCRIPT)
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


class RangeTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = bytes(range(256)) * 100 + b"last bytes"
        cls.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/ignore-range":
                    self.send_response(200)
                    self.end_headers()
                    return
                match = re.fullmatch(
                    r"bytes=(\d+)-(\d+)", self.headers.get("Range", "")
                )
                start, end = map(int, match.groups())
                cls.requests.append((start, end))
                body = cls.payload[start : end + 1]
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{end}/{len(cls.payload)}"
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_partial_range_resume_and_full_integrity(self):
        digest = hashlib.sha256(self.payload).hexdigest()
        with TemporaryDirectory() as directory:
            target = Path(directory) / "sample.zip"
            parts = target.parent / ".parts" / digest
            parts.mkdir(parents=True)
            (parts / "00000.part").write_bytes(self.payload[:1000])
            self.requests.clear()
            transport.download_ranges(
                self.url,
                {},
                target,
                len(self.payload),
                digest,
                workers=3,
                part_bytes=4096,
            )
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertIn((1000, 4095), self.requests)
            self.assertFalse(parts.exists())

    def test_server_ignoring_range_is_rejected(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "sample.zip"
            with self.assertRaisesRegex(RuntimeError, "Expected partial response"):
                transport.download_ranges(
                    self.url + "/ignore-range",
                    {},
                    target,
                    len(self.payload),
                    hashlib.sha256(self.payload).hexdigest(),
                    workers=1,
                    part_bytes=len(self.payload),
                )
            self.assertFalse(target.exists())

    def test_wrong_final_hash_never_publishes_target(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "sample.zip"
            with self.assertRaisesRegex(RuntimeError, "failed SHA256"):
                transport.download_ranges(
                    self.url,
                    {},
                    target,
                    len(self.payload),
                    "0" * 64,
                    workers=1,
                    part_bytes=len(self.payload),
                )
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
