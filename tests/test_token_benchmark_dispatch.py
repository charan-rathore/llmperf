import itertools
import threading
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import token_benchmark_ray as benchmark
from llmperf import common_metrics as metrics


class DispatchTest(unittest.TestCase):
    def test_slow_first_request_preserves_prompt_mix_and_request_budget(self):
        budget = 8
        sent = []
        lock = threading.Lock()
        release_first = threading.Event()
        first_saw_progress = []
        prompt_ids = itertools.count()

        class Launcher:
            def __init__(self, clients):
                pass

            def launch_requests(self, config):
                self.prompt_id = int(config.prompt[0])
                with lock:
                    self.first = len(sent) == 0
                    sent.append(self.prompt_id)

            def get_next_ready(self):
                if not hasattr(self, "first"):
                    return []
                if self.first:
                    first_saw_progress.append(release_first.wait(5))
                return [({
                    metrics.INTER_TOKEN_LAT: 1,
                    metrics.NUM_OUTPUT_TOKENS: 1,
                    metrics.NUM_INPUT_TOKENS: 1,
                    metrics.E2E_LAT: 1,
                }, "x", None)]

        completed = itertools.count(1)

        def record_completion(count):
            if next(completed) == budget - 1:
                release_first.set()

        with ExitStack() as mocks:
            mocks.enter_context(patch.object(benchmark, "RequestsLauncher", Launcher))
            mocks.enter_context(patch.object(benchmark, "construct_clients"))
            mocks.enter_context(patch.object(benchmark, "metrics_summary"))
            prompt = mocks.enter_context(
                patch.object(benchmark, "randomly_sample_sonnet_lines_prompt")
            )
            prompt.side_effect = lambda **kwargs: (str(next(prompt_ids)), 1)
            tokenizer = mocks.enter_context(patch.object(benchmark, "LlamaTokenizerFast"))
            tokenizer.from_pretrained.return_value.encode.side_effect = list
            progress = mocks.enter_context(patch.object(benchmark, "tqdm"))
            progress.return_value.update.side_effect = record_completion

            _, recorded = benchmark.get_token_throughput_latencies(
                "test", 1, 0, 1, 0,
                num_concurrent_requests=2, max_num_completed_requests=budget,
            )

        self.assertEqual(first_saw_progress, [True])
        self.assertEqual(sorted(sent), list(range(budget)))
        self.assertEqual(len(recorded), budget)
