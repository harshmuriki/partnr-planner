"""API-free billing tests; load leaf modules without importing Habitat."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


usage = load('cost_usage', 'habitat_llm/utils/llm_usage.py')
with patch.dict('sys.modules', {'habitat_llm.utils.llm_usage': usage}):
    costs = load('cost_report', 'habitat_llm/utils/llm_cost_report.py')


def payload(writes=700, cached=200, prompt=1000, output=50):
    details = {'cached_tokens': cached}
    if writes is not None:
        details['cache_write_tokens'] = writes
    return {'id': 'response-test', 'model': 'gpt-5.6-luna',
            'usage': {'prompt_tokens': prompt, 'completion_tokens': output,
                      'prompt_tokens_details': details,
                      'completion_tokens_details': {'reasoning_tokens': 30}},
            'choices': [{'message': {'role': 'assistant', 'content': 'Done[]'}, 'finish_reason': 'stop'}]}


class CostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.ledger = costs.EpisodeCostLedger()
        self.ledger.path = Path(self.temp.name) / 'cost.json'
        self.token = costs._ACTIVE_LEDGER.set(self.ledger)

    def tearDown(self):
        costs._ACTIVE_LEDGER.reset(self.token)
        os.chdir(self.cwd)
        self.temp.cleanup()

    def test_partition_reasoning_and_long_context(self):
        bill = costs.price_usage('gpt-5.6-luna', payload()['usage'])
        self.assertAlmostEqual(bill['total_usd'], (100*.2 + 200*.02 + 700*.25 + 50*1.2)/1e6)
        self.assertEqual(bill['tokens']['reasoning_output'], 30)
        self.assertEqual(bill['cost_usd']['output'], 50*1.2/1e6)
        for _ in range(2):
            with costs.track_model_call('gpt-5.6-luna'):
                costs.record_api_response(payload(prompt=200_000, writes=200_000, cached=0, output=0))
        self.assertAlmostEqual(costs.summarize_calls(self.ledger.calls)['total_usd'], .1)

    def test_missing_usage_is_not_free_and_missing_writes_have_bounds(self):
        with costs.track_model_call('gpt-5.6-luna'):
            costs.record_api_response(payload(writes=None))
        b = self.ledger.calls[0]['billing']
        self.assertIsNone(b['total_usd'])
        self.assertGreater(b['upper_bound_usd'], b['lower_bound_usd'])
        with self.assertRaises(TimeoutError):
            with costs.track_model_call('gpt-5.6-luna'):
                raise TimeoutError()
        summary = costs.summarize_calls(self.ledger.calls)
        self.assertIsNone(summary['total_usd'])
        self.assertIsNone(summary['upper_bound_usd'])
        self.assertEqual(len(json.loads(self.ledger.path.read_text())['calls']), 2)

    def test_unknown_model_tier_and_local_cache(self):
        self.assertIsNone(costs.price_usage('unknown', payload()['usage'])['total_usd'])
        self.assertIsNone(costs.price_usage('gpt-5.6-luna', payload()['usage'], 'priority')['total_usd'])
        costs.record_local_cache_hit('gpt-5.6-luna')
        summary = costs.summarize_calls(self.ledger.calls)
        self.assertEqual(summary['total_usd'], 0)
        self.assertEqual(summary['api_attempts'], 0)
        self.assertEqual(summary['local_cache_hits'], 1)

    def test_episode_continuation_reset_and_exception(self):
        class Runner:
            @costs.record_episode_costs
            def run(self, instruction=None, output_name='', preserve_planner_state=False):
                self._episode_cost_ledger.path = Path('episode.json')
                with costs.track_model_call('gpt-5.6-luna'):
                    costs.record_api_response(payload())
                if instruction == 'fail':
                    raise ValueError('failure after billable response')
                return {}
        runner = Runner()
        first = runner.run()['model_costs']['total_usd']
        self.assertAlmostEqual(runner.run(preserve_planner_state=True)['model_costs']['total_usd'], 2*first)
        self.assertEqual(runner.run()['model_costs']['total_usd'], first)
        with self.assertRaises(ValueError):
            runner.run('fail')
        data = json.loads(Path('episode.json').read_text())
        self.assertEqual(data['status'], 'interrupted')
        self.assertEqual(data['summary']['total_usd'], first)
        self.assertIs(costs._ACTIVE_LEDGER.get(), self.ledger)

    def test_offline_summary_prefers_request_ledger_and_keeps_unknown_attempt(self):
        with patch.dict('sys.modules'):
            summary_module = load('cost_summary', 'scripts/summarize_model_costs.py')
        root = Path(self.temp.name)
        run = root / 'react/T3-ACC-BASE/repeat_1/attempt_000/runs/run1'
        run.mkdir(parents=True)
        (run / 'run.json').write_text(json.dumps({'episode_id': 'T3-ACC-BASE', 'status': 'completed',
                                               'metrics': {'cost_metrics': {'prompt_tokens': 999999}}}))
        ledger = costs.EpisodeCostLedger()
        ledger.path = run / 'dataset/llm_costs/task.json'
        ledger.metadata = {'episode_id': 'T3-ACC-BASE', 'baseline': 'react'}
        ledger.status = 'completed'
        token = costs._ACTIVE_LEDGER.set(ledger)
        try:
            with costs.track_model_call('gpt-5.6-luna'):
                costs.record_api_response(payload())
        finally:
            costs._ACTIVE_LEDGER.reset(token)
        rows = summary_module.collect(root, 'T3-ACC-BASE')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['kind'], 'request_ledger')
        self.assertEqual(summary_module.aggregate(rows)[0]['total_usd'], ledger.calls[0]['billing']['total_usd'])
        missing = root / 'react/T3-ACC-BASE/repeat_1/attempt_001/runs/run1'
        missing.mkdir(parents=True)
        (missing / 'run.json').write_text(json.dumps({'episode_id': 'T3-ACC-BASE', 'status': 'timeout', 'metrics': {}}))
        rows = summary_module.collect(root, 'T3-ACC-BASE')
        self.assertEqual(len(rows), 2)
        self.assertIsNone(summary_module.aggregate(rows)[0]['total_usd'])
        self.assertIsNone(summary_module.aggregate(rows)[0]['recorded_usage_upper_usd'])

    def test_real_openai_sdk_retries_and_wrapper_accounting(self):
        import httpx
        from openai import OpenAI
        from omegaconf import OmegaConf
        stubs = {'habitat_llm.utils.llm_usage': usage, 'habitat_llm.utils.llm_cost_report': costs,
                 'habitat_llm.llm.base_llm': SimpleNamespace(BaseLLM=object, Prompt=object)}
        with patch.dict('sys.modules', stubs):
            module = load('cost_openai_chat', 'habitat_llm/llm/openai_chat.py')
        responses = iter([httpx.Response(500, json={'error': {'message': 'retry'}}),
                          httpx.Response(200, json=payload())])
        client = httpx.Client(transport=httpx.MockTransport(lambda r: next(responses)),
                             event_hooks={'request': [costs.billing_http_request], 'response': [costs.billing_http_response]})
        llm = module.OpenAIChat.__new__(module.OpenAIChat)
        llm.client = OpenAI(api_key='not-a-real-key', http_client=client, max_retries=1)
        llm.generation_params = OmegaConf.create({'model': 'gpt-5.6-luna', 'stop': ''})
        llm.llm_conf = SimpleNamespace(system_message='Test')
        llm.message_history = []
        llm.keep_message_history = False
        llm.token_usage = usage.TokenUsageTracker('gpt-5.6-luna')
        self.assertEqual(llm.generate('test'), 'Done[]')
        self.assertEqual(len(self.ledger.calls), 2)
        self.assertEqual(self.ledger.calls[1]['sdk_attempt'], 2)
        self.assertEqual(self.ledger.calls[1]['http_status'], 200)
        self.assertTrue(self.ledger.calls[1]['billing']['complete'])
        self.assertEqual(llm.token_usage.calls, 1)
        self.assertIsNone(costs.summarize_calls(self.ledger.calls)['total_usd'])
        client.close()

    def test_vlm_invalid_response_usage_and_local_replay(self):
        stubs = {'habitat_llm.utils.llm_usage': usage, 'habitat_llm.utils.llm_cost_report': costs}
        with patch.dict('sys.modules', stubs):
            module = load('cost_vlm_api', 'habitat_llm/vlm_tamp/vlm_api.py')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'not-a-real-key'}):
            vlm = module.GPT4vApi(cache_path='cache.json')
        invalid = payload()
        invalid['choices'] = []
        response = SimpleNamespace(status_code=200, headers={}, json=lambda: invalid)
        with patch.object(module.requests, 'post', return_value=response):
            self.assertEqual(vlm.ask('bad', use_cache=False), '')
        self.assertEqual(vlm.token_usage.calls, 1)
        self.assertTrue(self.ledger.calls[0]['billing']['complete'])
        response.json = lambda: payload()
        with patch.object(module.requests, 'post', return_value=response) as post:
            self.assertEqual(vlm.ask('good'), 'Done[]')
            self.assertEqual(vlm.ask('good'), 'Done[]')
            self.assertEqual(post.call_count, 1)
        summary = costs.summarize_calls(self.ledger.calls)
        self.assertEqual(summary['api_attempts'], 2)
        self.assertEqual(summary['local_cache_hits'], 1)
        self.assertAlmostEqual(summary['total_usd'], 2*self.ledger.calls[0]['billing']['total_usd'])


if __name__ == '__main__':
    unittest.main()
