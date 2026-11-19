"""Durable episode-wide model billing records, independent of planner resets.

Only usage and request identifiers are persisted, never prompts or credentials.
Dollar amounts are public API list-price calculations, not account invoices.
"""
import contextvars
import functools
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from habitat_llm.utils.llm_usage import prices_for_model, GPT56_LONG_CONTEXT_TOKENS, LONG_CONTEXT_SURCHARGE_PREFIXES

_ACTIVE_LEDGER = contextvars.ContextVar('episode_cost_ledger', default=None)
_ACTIVE_CALL = contextvars.ContextVar('episode_model_call', default=None)
PRICING_DATE = '2026-10-03'
PRICING_SOURCES = [
    'https://developers.openai.com/api/docs/models/gpt-5.6-luna',
    'https://developers.openai.com/api/docs/guides/prompt-caching',
    'https://developers.openai.com/api/docs/pricing',
]
COMPONENTS = ('ordinary_input', 'cache_read', 'cache_write', 'output')


def _dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    return vars(value) if hasattr(value, '__dict__') else {}


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def price_usage(model, usage, service_tier='default', provider='openai'):
    """Price one request; missing cache fields produce bounds, never fake zeros."""
    usage = _dict(usage)
    details = _dict(usage.get('prompt_tokens_details') or usage.get('input_tokens_details'))
    output_details = _dict(usage.get('completion_tokens_details') or usage.get('output_tokens_details'))
    prompt = _count(usage.get('prompt_tokens', usage.get('input_tokens')))
    output = _count(usage.get('completion_tokens', usage.get('output_tokens')))
    cached = _count(details.get('cached_tokens'))
    writes = _count(details.get('cache_write_tokens'))
    rates = prices_for_model(model) if provider == 'openai' else None
    rates = dict(rates) if rates else None
    issues = []
    if rates is None:
        issues.append('unknown_model_pricing')
    if service_tier not in (None, 'default', 'standard'):
        issues.append('unsupported_service_tier')
    if prompt is None or output is None:
        issues.append('missing_usage')
    if rates and prompt is not None and prompt > GPT56_LONG_CONTEXT_TOKENS and model.lower().startswith(LONG_CONTEXT_SURCHARGE_PREFIXES):
        rates = {k: v * (1.5 if k == 'output' else 2) for k, v in rates.items()}
    if rates and 'cache_write' not in rates:
        # Older models do not bill a separate cache-write category.
        writes = 0
    if prompt == 0:
        cached, writes = 0, 0
    if cached is None:
        issues.append('missing_cache_read_tokens')
    if writes is None:
        issues.append('missing_cache_write_tokens')
    if prompt is not None and (cached or 0) + (writes or 0) > prompt:
        issues.append('inconsistent_usage')
    ordinary = prompt - cached - writes if all(v is not None for v in (prompt, cached, writes)) else None
    tokens = {'input': prompt, 'ordinary_input': ordinary, 'cache_read': cached,
              'cache_write': writes, 'output': output,
              'reasoning_output': _count(output_details.get('reasoning_tokens'))}
    costs = dict.fromkeys(COMPONENTS)
    low = high = None
    if rates and prompt is not None and output is not None and not any(
        i in issues for i in ('unsupported_service_tier', 'inconsistent_usage')
    ):
        prices = {'ordinary_input': rates['input'], 'cache_read': rates.get('cached', rates['input']),
                  'cache_write': rates.get('cache_write', rates['input']), 'output': rates['output']}
        costs = {k: tokens[k] * prices[k] / 1e6 if tokens[k] is not None else None for k in COMPONENTS}
        known = (cached or 0) * prices['cache_read'] + (writes or 0) * prices['cache_write'] + output * prices['output']
        remaining = prompt - (cached or 0) - (writes or 0)
        possible = [prices['ordinary_input']]
        if cached is None:
            possible.append(prices['cache_read'])
        if writes is None:
            possible.append(prices['cache_write'])
        low = (known + remaining * min(possible)) / 1e6
        high = (known + remaining * max(possible)) / 1e6
    return {'tokens': tokens, 'rates_usd_per_million': rates, 'cost_usd': costs,
            'total_usd': low if not issues else None,
            'lower_bound_usd': low, 'upper_bound_usd': high,
            'complete': not issues, 'issues': issues}


def summarize_calls(calls):
    priced = [c['billing'] for c in calls]
    def total(key):
        values = [p[key] for p in priced]
        return sum(values) if all(v is not None for v in values) else None
    return {
        'api_attempts': sum(c['status'] != 'local_cache_hit' for c in calls),
        'local_cache_hits': sum(c['status'] == 'local_cache_hit' for c in calls),
        'complete': all(p['complete'] for p in priced),
        'total_usd': total('total_usd'),
        'known_cost_subtotal_usd': sum(p['total_usd'] or 0 for p in priced),
        'lower_bound_usd': sum(p['lower_bound_usd'] or 0 for p in priced),
        'upper_bound_usd': total('upper_bound_usd'),
        'cost_usd': {k: sum(p['cost_usd'][k] for p in priced)
                     if all(p['cost_usd'][k] is not None for p in priced) else None for k in COMPONENTS},
        'tokens': {k: sum(p['tokens'][k] for p in priced)
                   if all(p['tokens'][k] is not None for p in priced) else None
                   for k in ('input', 'ordinary_input', 'cache_read', 'cache_write', 'output', 'reasoning_output')},
        'issues': sorted({issue for p in priced for issue in p['issues']}),
    }


class EpisodeCostLedger:
    def __init__(self):
        self.path = None
        self.metadata = {}
        self.calls = []
        self.status = 'running'

    def bind(self, runner):
        if self.path is not None:
            # One ledger per episode even if a continuing subgoal changes its name.
            self.persist()
            return
        self.path = Path(runner.output_dir) / 'llm_costs' / (Path(runner.episode_filename).name + '.json')
        episode = runner.env_interface.env.env.env._env.current_episode
        planner_configs = getattr(runner.evaluation_runner_config, 'agents', {})
        planners = {str(uid): str(agent.planner._target_) for uid, agent in planner_configs.items()}
        labels = []
        for agent in planner_configs.values():
            name = str(agent.planner._target_).rsplit('.', 1)[-1]
            plan = agent.planner.plan_config
            if name == 'VlmTampPddlPlanner':
                label = 'vlm-tamp' if getattr(plan.vlm, 'use_images', True) else 'vlm-tamp-no-image'
            elif name == 'ZeroShotReactPlanner':
                label = 'react' if getattr(plan, 'send_action_image', False) else 'react-no-image'
            else:
                label = 'tru-pomdp' if name == 'TruPOMDPPlanner' else name
            labels.append(label)
        self.metadata = {'baseline': ','.join(labels), 'planner_targets': planners,
                         'episode_name': runner.episode_filename,
                         'episode_id': str(episode.episode_id),
                         'instruction': getattr(episode, 'instruction', runner.current_instruction)}
        self.persist()

    def persist(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {'schema_version': 1, **self.metadata, 'status': self.status,
                    'pricing_date': PRICING_DATE, 'pricing_sources': PRICING_SOURCES,
                    'billing_basis': 'Public standard API list prices; excludes account discounts, credits, taxes and non-model compute.',
                    'token_note': 'Input includes billed image tokens. Output includes reasoning tokens; do not add either again.',
                    'summary': summarize_calls(self.calls), 'calls': self.calls}
        tmp = self.path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(document, indent=2) + '\n')
        tmp.replace(self.path)

    def start(self, model, provider, source):
        entry = {'sequence': len(self.calls) + 1, 'started_at': datetime.now(timezone.utc).isoformat(),
                 'model': model, 'provider': provider, 'source': source, 'status': 'pending',
                 'billing': price_usage(model, {}, provider=provider)}
        self.calls.append(entry)
        self.persist()
        return entry


class ModelCall:
    def __init__(self, ledger, model, provider, source):
        self.ledger = ledger
        self.entry = ledger.start(model, provider, source)
        self.http_requests = 0

    def response(self, payload, http_status=None, request_id=None):
        payload = _dict(payload)
        usage = _dict(payload.get('usage'))
        self.entry.update(model=payload.get('model') or self.entry['model'],
                          response_id=payload.get('id'), request_id=request_id,
                          http_status=http_status, service_tier=payload.get('service_tier') or 'default',
                          usage=usage, status='response' if usage else 'response_without_usage')
        self.entry['billing'] = price_usage(self.entry['model'], usage, self.entry['service_tier'], self.entry['provider'])
        self.ledger.persist()


@contextmanager
def track_model_call(model, provider='openai', source='llm'):
    ledger = _ACTIVE_LEDGER.get()
    call = ModelCall(ledger, model, provider, source) if ledger else None
    token = _ACTIVE_CALL.set(call)
    try:
        yield call
    except BaseException as exc:
        if call:
            call.entry['error_type'] = type(exc).__name__
            if call.entry['status'] == 'pending':
                call.entry['status'] = 'request_error'
        raise
    finally:
        if call:
            if call.entry['status'] == 'pending':
                call.entry['status'] = 'no_response_usage'
            ledger.persist()
        _ACTIVE_CALL.reset(token)


def record_api_response(payload, http_status=None, request_id=None):
    call = _ACTIVE_CALL.get()
    if call:
        call.response(payload, http_status, request_id)


def record_local_cache_hit(model):
    ledger = _ACTIVE_LEDGER.get()
    if ledger:
        entry = ledger.start(model, 'local', 'vlm')
        entry['status'] = 'local_cache_hit'
        entry['billing'] = {'tokens': {k: 0 for k in ('input', *COMPONENTS, 'reasoning_output')},
                            'rates_usd_per_million': None, 'cost_usd': {k: 0.0 for k in COMPONENTS},
                            'total_usd': 0.0, 'lower_bound_usd': 0.0, 'upper_bound_usd': 0.0,
                            'complete': True, 'issues': []}
        ledger.persist()


def billing_http_request(request):
    """Observe every OpenAI SDK HTTP attempt, including automatic retries."""
    call = _ACTIVE_CALL.get()
    if call:
        if call.http_requests:
            if call.entry['status'] == 'pending':
                call.entry['status'] = 'no_response_usage'
            call.entry = call.ledger.start(call.entry['model'], call.entry['provider'], call.entry['source'])
        call.http_requests += 1
        call.entry['sdk_attempt'] = call.http_requests
        call.ledger.persist()


def billing_http_response(response):
    if _ACTIVE_CALL.get():
        response.read()  # OpenAIChat explicitly rejects streaming.
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        record_api_response(payload, response.status_code, response.headers.get('x-request-id'))


def bind_episode_cost_report(runner):
    ledger = _ACTIVE_LEDGER.get()
    if ledger:
        ledger.bind(runner)


def record_episode_costs(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        preserve = kwargs.get('preserve_planner_state', args[2] if len(args) > 2 else False)
        ledger = getattr(self, '_episode_cost_ledger', None) if preserve else None
        ledger = ledger or EpisodeCostLedger()
        self._episode_cost_ledger = ledger
        ledger.status = 'running'
        token = _ACTIVE_LEDGER.set(ledger)
        try:
            result = method(self, *args, **kwargs)
            ledger.status = 'completed'
            result['model_costs'] = summarize_calls(ledger.calls)
            result['model_cost_report'] = str(ledger.path)
            return result
        except BaseException:
            ledger.status = 'interrupted'
            raise
        finally:
            try:
                ledger.persist()
                if ledger.path:
                    summary = summarize_calls(ledger.calls)
                    amount = summary['total_usd']
                    display = f'${amount:.8f}' if amount is not None else 'incomplete (see report)'
                    print(f'Total LLM/VLM spend: {display}; report: {ledger.path}')
            finally:
                _ACTIVE_LEDGER.reset(token)
    return wrapped
