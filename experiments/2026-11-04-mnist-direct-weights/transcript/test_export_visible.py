"""Small deterministic tests for the publication export's boundaries."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("export_visible", Path(__file__).with_name("export_visible.py"))
export_visible = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_visible)


class VisibleExportTests(unittest.TestCase):
    def test_role_allowlist_and_actual_visible_text(self):
        messages, metadata = export_visible.export([
            {"role": "system", "text": "private system instructions"},
            {"role": "developer", "text": "private developer instructions"},
            {"role": "reasoning", "text": "private analysis"},
            {"role": "tool", "text": "private tool output"},
            {"role": "user", "text": "Write numerical weights", "timestamp": "2026-11-04T00:00:00Z"},
            {"role": "assistant", "text": "Measured 96.03%", "timestamp": "2026-11-04T01:00:00Z"},
        ])
        self.assertEqual([m["text"] for m in messages], ["Write numerical weights", "Measured 96.03%"])
        self.assertEqual([m["message"] for m in messages], [1, 2])
        self.assertEqual(metadata["published_messages"], 2)
        self.assertEqual(set(metadata["dropped_input_roles"]), {"system", "developer", "reasoning", "tool"})

    def test_private_path_and_context_replacement(self):
        home_prefix = "/ho" + "me/example"
        source = ("<environment_context>private metadata</environment_context> "
                  + home_prefix + "/dev/mnist-astra-direct-weights/model.py "
                  "/root/integrity_continued")
        text, changes = export_visible.sanitize(source)
        self.assertIn("<workspace>/model.py", text)
        self.assertIn("reviewer-1", text)
        self.assertNotIn("private metadata", text)
        self.assertNotIn(home_prefix, text)
        self.assertEqual(changes["injected_context_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
