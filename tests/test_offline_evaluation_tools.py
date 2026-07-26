import hashlib
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import evaluate_sr
import simulate_draw


class _Response:
    def __init__(self, payload):
        self._stream = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self._stream.read(size)


class TestSuperResolutionEvaluator(unittest.TestCase):
    def test_download_is_verified_and_atomically_published(self):
        payload = b"verified model bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "model.bin")
            with mock.patch(
                "evaluate_sr.urllib.request.urlopen",
                return_value=_Response(payload),
            ) as urlopen:
                self.assertTrue(
                    evaluate_sr.download_file(
                        "https://example.invalid/model.bin",
                        path,
                        len(payload),
                        digest,
                        timeout_sec=1.5,
                    )
                )

            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), payload)
            self.assertFalse(os.path.exists(path + ".part"))
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 1.5)

    def test_failed_verification_preserves_existing_cache(self):
        original = b"existing invalid cache"
        payload = b"unexpected response"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "model.bin")
            with open(path, "wb") as stream:
                stream.write(original)
            with mock.patch(
                "evaluate_sr.urllib.request.urlopen",
                return_value=_Response(payload),
            ):
                self.assertFalse(
                    evaluate_sr.download_file(
                        "https://example.invalid/model.bin",
                        path,
                        len(payload),
                        hashlib.sha256(b"different").hexdigest(),
                    )
                )

            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), original)
            self.assertFalse(os.path.exists(path + ".part"))

    def test_help_exits_before_asset_download(self):
        with (
            mock.patch.object(evaluate_sr, "download_file") as download,
            redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            evaluate_sr.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        download.assert_not_called()


class TestDrawSimulator(unittest.TestCase):
    def test_fixed_variant_drives_real_draw_mode(self):
        gate = simulate_draw.make_variant("fixed")
        self.assertEqual(type(gate.mode).__name__, "DrawMode")

    def test_short_synthetic_run_succeeds(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(simulate_draw.main(["--seconds", "1"]), 0)

    def test_invalid_replay_returns_nonzero(self):
        with (
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                simulate_draw.main(["--replay", "missing-trace.jsonl"]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
