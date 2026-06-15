import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.frame_recorder import FrameRecorder


class TestFrameRecorder(unittest.TestCase):
    def test_async_recorder_flushes_metadata_on_close(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = FrameRecorder(
                out_root=temp_dir,
                max_frames=3,
                max_seconds=10,
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            for _ in range(3):
                recorder.write(frame)
            recorder.close()

            meta_path = os.path.join(recorder.dir, "meta.jsonl")
            with open(meta_path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["i"] for row in rows], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
