"""CPU-only helper tests. Never connects to SSH, Docker, or GPUs."""
import base64
from contextlib import nullcontext
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import tempfile
import threading
import unittest

from remote_exec import build_command
from render_launch import render
from reserve_vram import allocate, reservation_bytes
from verify_service import DEFAULT_CASES, verify


ROOT = Path(__file__).resolve().parents[1]


class TransportTests(unittest.TestCase):
    def test_unicode_and_shell_text_survive_transport(self):
        script = '\ufeff# 模型\r\necho "$HOME $(hostname)"\r\n'
        command = build_command("hcu-node", script)
        remote = shlex.split(command[-1])
        decoded = base64.b64decode(remote[2]).decode("utf-8")
        self.assertEqual(decoded, '# 模型\necho "$HOME $(hostname)"\n')
        self.assertNotIn("BatchMode=yes", command)

    def test_container_is_one_quoted_argument(self):
        command = build_command("hcu-node", "true", "name; echo injected", "/home/a b")
        parts = shlex.split(command[-1])
        self.assertEqual(parts[-4:], ["/home/a b", "name; echo injected", "bash", "-s"])

    def test_ssh_option_injection_is_rejected(self):
        for host in ("-oProxyCommand=bad", "hcu-node;evil", "hcu-node -p 22"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                build_command(host, "true")


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / "references/qwen38-tp4-example.json").read_text(encoding="utf-8"))

    def test_known_profile_has_expected_effective_arguments(self):
        text = render(self.profile)
        command = shlex.split(text.splitlines()[-1])[1:]
        self.assertEqual(command[command.index("--tp-size") + 1], "4")
        self.assertEqual(command[command.index("--page-size") + 1], "64")
        self.assertEqual(command[command.index("--attention-backend") + 1], "fa3")
        self.assertIn("export HIP_VISIBLE_DEVICES=0,1,2,3", text)
        self.assertNotIn("TOKENIZERS_PARALLELISM", text)

    def test_shell_characters_in_paths_are_literal_arguments(self):
        self.profile["model_path"] = "/models/a b/$(echo literal)"
        command = shlex.split(render(self.profile).splitlines()[-1])[1:]
        self.assertEqual(command[command.index("--model-path") + 1], self.profile["model_path"])

    def test_conflicting_or_invalid_inputs_fail(self):
        edits = [
            {"devices": [0, 0]}, {"devices": [True]}, {"port": 0},
            {"env": {"ROCR_VISIBLE_DEVICES": "1"}},
            {"launch_args": ["--model-path=/other"]},
        ]
        for edit in edits:
            profile = deepcopy(self.profile)
            profile.update(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                render(profile)


class ReservationTests(unittest.TestCase):
    def test_budget_respects_total_fraction_and_free_headroom(self):
        self.assertEqual(reservation_bytes(100, 100, .9, 2), 90)
        self.assertEqual(reservation_bytes(91, 100, .9, 5), 86)
        with self.assertRaises(RuntimeError):
            reservation_bytes(50, 100, .9, 2)

    def test_chunks_do_not_overallocate_and_stop_is_observed(self):
        stop = threading.Event()
        class FakeCUDA:
            def device(self, index):
                return nullcontext()
            def synchronize(self, index):
                pass
        class FakeBlock:
            def fill_(self, value):
                if torch.stop_after_first:
                    stop.set()
        class FakeTorch:
            cuda = FakeCUDA()
            uint8 = object()
            stop_after_first = False
            def __init__(self):
                self.sizes = []
            def empty(self, size, **kwargs):
                self.sizes.append(size)
                return FakeBlock()
        torch = FakeTorch()
        blocks = []
        self.assertEqual(allocate(torch, 0, 25, 10, stop, blocks), 25)
        self.assertEqual(torch.sizes, [10, 10, 5])
        self.assertEqual(len(blocks), 3)
        torch.sizes.clear()
        torch.stop_after_first = True
        self.assertEqual(allocate(torch, 0, 25, 10, stop, []), 10)
        self.assertEqual(torch.sizes, [10])


class ServiceTests(unittest.TestCase):
    def run_endpoint(self, listed="test-model", response_model="test-model", answer="323",
                     finish="stop", health_delay=False, all_cases=False):
        state = {"health_calls": 0, "chat_calls": 0}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def send_json(self, status, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_GET(self):
                if self.path == "/health":
                    state["health_calls"] += 1
                    code = 503 if health_delay and state["health_calls"] == 1 else 200
                    self.send_json(code, {})
                elif self.path == "/v1/models":
                    self.send_json(200, {"data": [{"id": listed}]})
                else:
                    self.send_json(404, {})
            def do_POST(self):
                state["chat_calls"] += 1
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if all_cases:
                    prompt = request["messages"][0]["content"]
                    text = next(c["expected"] for c in DEFAULT_CASES if c["prompt"] == prompt)
                else:
                    text = answer
                self.send_json(200, {"model": response_model, "choices": [{
                    "message": {"content": text}, "finish_reason": finish}]})
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "evidence.json"
                cases = None if all_cases else [DEFAULT_CASES[0]]
                result = verify(f"http://127.0.0.1:{server.server_port}/v1", "test-model", output,
                                cases=cases, wait_seconds=1, request_timeout=2, poll_seconds=.01)
                self.assertEqual(json.loads(output.read_text()), result)
                return result, state
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_health_wait_then_real_three_case_success(self):
        result, state = self.run_endpoint(health_delay=True, all_cases=True)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(state["health_calls"], 2)
        self.assertEqual(state["chat_calls"], 3)

    def test_wrong_identity_prevents_generation(self):
        result, state = self.run_endpoint(listed="other-model")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(state["chat_calls"], 0)

    def test_numeric_substring_is_not_a_correct_answer(self):
        result, _ = self.run_endpoint(answer="It could be 323 or 324")
        self.assertEqual(result["status"], "failed")

    def test_truncated_correct_text_is_not_a_pass(self):
        result, _ = self.run_endpoint(finish="length")
        self.assertEqual(result["status"], "failed")

    def test_response_model_mismatch_fails(self):
        result, _ = self.run_endpoint(response_model="other-model")
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
