"""Offline contracts: real loopback HTTP, synthetic adapters, no model claims."""

import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path

from heretic.vllm_backend import VllmBackend, VllmBackendError


class VllmBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp")
        self.root = Path(self.tmp.name)
        self.events = []
        self.loaded_weights = []
        self.fail_generation = False
        self.fail_unload = False
        self.invalid_indices = False
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.events.append((self.path, data))
                if self.path.endswith("load_lora_adapter") and not self.path.endswith(
                    "unload_lora_adapter"
                ):
                    path = Path(data["lora_path"])
                    owner.assertTrue((path / "adapter_model.safetensors").exists())
                    owner.loaded_weights.append(
                        (path / "adapter_model.safetensors").read_bytes()
                    )
                    body = b"loaded"
                elif self.path.endswith("unload_lora_adapter"):
                    if owner.fail_unload:
                        self.send_response(500)
                        self.end_headers()
                        return
                    body = b"unloaded"
                else:
                    if owner.fail_generation:
                        self.send_response(500)
                        self.end_headers()
                        return
                    choices = [
                        {"index": i, "token_ids": [tokens[0] + 10, 2]}
                        for i, tokens in enumerate(data["prompt"])
                    ]
                    if owner.invalid_indices:
                        choices[-1]["index"] = 0
                    body = json.dumps({"choices": list(reversed(choices))}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.backend = VllmBackend(
            f"http://127.0.0.1:{self.server.server_port}/v1",
            self.root,
            str(self.root),
            timeout=2,
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def export(self, path):
        (path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "fixture", "peft_type": "LORA"})
        )
        (path / "adapter_model.safetensors").write_bytes(b"synthetic test adapter")

    def generate(self):
        return self.backend.generate(
            [[1, 3], [2, 3]],
            self.export,
            base_model="fixture",
            max_tokens=10,
            eos_token_ids=[2],
            batch_size=2,
        )

    def test_fresh_snapshots_and_cache_salts_preserve_response_order(self):
        self.assertEqual(self.generate(), [[11, 2], [12, 2]])
        self.assertEqual(self.generate(), [[11, 2], [12, 2]])
        loads = [p for path, p in self.events if path.endswith("/load_lora_adapter")]
        completions = [p for path, p in self.events if path.endswith("/completions")]
        self.assertNotEqual(loads[0]["lora_name"], loads[1]["lora_name"])
        self.assertNotEqual(loads[0]["lora_path"], loads[1]["lora_path"])
        self.assertEqual(completions[0]["model"], loads[0]["lora_name"])
        self.assertNotEqual(completions[0]["cache_salt"], completions[1]["cache_salt"])
        self.assertEqual(completions[0]["temperature"], 0)
        self.assertTrue(completions[0]["return_token_ids"])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_generation_failure_unloads_and_never_falls_back(self):
        self.fail_generation = True
        with self.assertRaises(VllmBackendError):
            self.generate()
        self.assertTrue(self.events[-1][0].endswith("/unload_lora_adapter"))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unload_failure_preserves_snapshot_and_poisons_backend(self):
        self.fail_unload = True
        with self.assertRaises(VllmBackendError):
            self.generate()
        self.assertEqual(len(list(self.root.iterdir())), 1)
        count = len(self.events)
        with self.assertRaises(VllmBackendError):
            self.generate()
        self.assertEqual(len(self.events), count)

    def test_duplicate_response_indices_fail_and_cleanup(self):
        self.invalid_indices = True
        with self.assertRaises(VllmBackendError):
            self.generate()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_wrong_base_is_rejected_before_any_request(self):
        with self.assertRaises(VllmBackendError):
            self.backend.generate(
                [[1]],
                self.export,
                base_model="other",
                max_tokens=10,
                eos_token_ids=[2],
                batch_size=2,
            )
        self.assertEqual(self.events, [])
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
