"""Prepare and review image-ReAct, image-VLM-TAMP and TRU-POMDP annotations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import copy
import fcntl
import io
import json
import math
import os
from pathlib import Path
import time
import uuid

from judge_contract import HERE, LABELS, FAILURES, INEFFICIENCIES, SCHEMA, INPUT_SCHEMA, SYSTEM, VERSION, CHECKPOINT_POLICY, TIMEOUT_POLICY, validate, validate_packet
from judge_evidence import BASELINES, FINISHED, catalog, load, packet_for, sha
from study import atomic_json, now
from judge_prompt import PROMPT_POLICY
from judge_html_input import build_input, validation_view, VERSION as HTML_INPUT_VERSION
from judge_counts import annotation_counts

ANNOTATION_BASELINES = ('react', 'vlm-tamp', 'tru-pomdp')


def check_annotation_scope(baseline=None, run_id=None):
    selected = [value for value in (baseline, run_id.split('/')[0] if run_id else None) if value]
    if any(value not in ANNOTATION_BASELINES for value in selected):
        raise ValueError('Annotation is restricted to image-enabled react, image-enabled vlm-tamp, and tru-pomdp; no-image baselines are excluded')


def example_set(directory):
    """Only explicitly enabled, reviewed examples enter requests; templates are disabled."""
    examples = []
    for p in sorted(Path(directory).glob('*.json')):
        item = load(p)
        if not item.get('enabled', False):
            continue
        if not item.get('reviewed_by') or not item.get('input') or 'expected_output' not in item:
            raise ValueError(f'{p}: enabled examples need reviewer, input and expected_output')
        item = copy.deepcopy(item)
        # Older reviewed examples predate the additive timeout category. Do not
        # invent a reviewed negative: mark the new category explicitly unassessed.
        assessments = item['expected_output'].get('label_assessments', [])
        if 'timeout' in LABELS and not any(a['label'] == 'timeout' for a in assessments):
            assessments.append({'label': 'timeout',
                'status': 'not_applicable' if item['input']['simulator']['outcome'] in ('complete', 'unavailable') else 'insufficient_evidence',
                'reason': 'Legacy example predates timeout review; no timeout diagnosis is asserted.'})
        for family in ['failures', 'inefficiencies']:
            for incident in item['expected_output'].get(family, []):
                incident.pop('confidence_calibrated', None)
                incident.pop('review_status', None)
        shown = validation_view(item['model_input'], item['input']) if item.get('model_input') else item['input']
        validate(item['expected_output'], shown)
        examples.append({'name': p.name, 'input': item['input'], 'expected_output': item['expected_output'],
                         'model_input': item.get('model_input'),
                         'source_run_ids': item.get('source_run_ids', [])})
    return examples


def prepare(source, out, limit=0, baseline=None, run_id=None, overrides=None):
    check_annotation_scope(baseline, run_id)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    if overrides is None and manifest.exists():
        overrides = load(manifest).get('overrides')
    overrides = str(Path(overrides).resolve()) if overrides is not None else None
    runs = catalog(source, overrides)
    if manifest.exists() and load(manifest)['source'] != str(Path(source).resolve()):
        raise ValueError('Output directory belongs to another source')
    atomic_json(manifest, {'version': VERSION, 'source': str(Path(source).resolve()), 'overrides': overrides, 'updated': now(), 'runs': runs})
    for name, source_file in [('rubric.md', HERE / 'astra_rubric.md'), ('system_prompt.txt', None)]:
        p = out / name
        if not p.exists():
            p.write_text(source_file.read_text() if source_file else SYSTEM)
    atomic_json(out / 'output.schema.json', SCHEMA)
    atomic_json(out / 'input.schema.json', INPUT_SCHEMA)
    (out / 'index.html').write_text((HERE / 'templates/run_judge.html').read_text())
    examples = out / 'examples'
    examples.mkdir(exist_ok=True)
    if not (examples / 'template.json').exists():
        atomic_json(examples / 'template.json', {'enabled': False, 'reviewed_by': '', 'source_run_ids': [],
                    'input': {}, 'expected_output': {},
                    'instructions': 'Copy a packets/*.json and a corrected analyses/*.json analysis here. Remove provenance from input if desired. Enable only after human review.'})
    count, states = 0, Counter()
    for run in runs:
        if run['baseline'] not in ANNOTATION_BASELINES:
            continue
        if baseline and run['baseline'] != baseline or run_id and run['id'] != run_id:
            continue
        if run['status'] not in FINISHED or not run['path']:
            states['deferred'] += 1
            continue
        if limit and count >= limit:
            break
        count += 1
        try:
            packet = validate_packet(packet_for(run))
            atomic_json(out / 'packets' / (run['id'] + '.json'), packet)
            states['prepared'] += 1
        except Exception as exc:
            states['error'] += 1
            atomic_json(out / 'errors' / (run['id'] + '.json'), {'stage': 'prepare', 'error': str(exc), 'at': now()})
        print(run['id'], dict(states), flush=True)
    atomic_json(out / 'prepare_status.json', {'updated': now(), 'counts': dict(states)})
    return dict(states)


def request_for(out, packet, model='gpt-6-astra', effort='high', max_input_tokens=950000, images=None):
    if images or packet.get('image_inputs'):
        raise ValueError('Astra annotation is text-only; image inputs are not supported')
    validate_packet(packet)
    out = Path(out)
    examples = example_set(out / 'examples')
    rid = packet['provenance']['run_id']
    check_annotation_scope(run_id=rid)
    if any(rid in e['source_run_ids'] for e in examples):
        raise ValueError('Run is a prompt example; exclude it from held-out judging')
    # Model identity, baseline and repetition are retained in provenance, not shown to the judge.
    for example in examples:
        if example['input'].get('image_inputs'):
            raise ValueError('Astra annotation is text-only; examples must use text/data evidence')
    body, html_hashes = build_input(packet, (out / 'rubric.md').read_text(), examples)
    text = json.dumps(body, ensure_ascii=False, separators=(',', ':'))
    system = (out / 'system_prompt.txt').read_text()
    # Raw packet decoding instructions do not apply to the readable model view.
    system = system.replace(CHECKPOINT_POLICY, '')
    system = system.replace('Timeout alone supplies no failure cause.',
                            'A recorded execution cutoff can support timeout; it does not establish a planning failure.')
    if TIMEOUT_POLICY not in system:
        system += '\n' + TIMEOUT_POLICY
    system = system.replace('Expand {"json_ref":"jN"} through json_blocks (possibly nested). Expand\n'
                            'text_block_sequence by concatenating strings and {"ref":"bN"} from text_blocks.\n'
                            'These are lossless deduplication, not missing observations.\n', '')
    # Preserve behavioral guidance, but describe only the new displayed format.
    behavioral = PROMPT_POLICY[PROMPT_POLICY.index('Labels are not mutually exclusive.'):PROMPT_POLICY.index('In simulator.criteria')]
    system += '\n' + behavioral + '\n' + (
        'No images or video frames are supplied. Use only text/data evidence. '
        'Input has scenario, initial_scene, initial_robot_memory, trace, subtask_results, '
        'definitions, examples and output_format. Steps are HTML high-level attempts; '
        'step IDs are also their evidence IDs. Separate plan IDs are planning context, '
        'not additional actions. Cite only supplied IDs. State snapshots are reference '
        'truth, not robot knowledge. Subtask results are immutable; checkpoint scores '
        'cannot authorize failure labels. Examples are partial demonstrations, not '
        'negative labels. Numeric incident counts are derived by code after validation.')
    import tiktoken
    tokens = len(tiktoken.get_encoding('o200k_base').encode(system + text + json.dumps(SCHEMA), disallowed_special=()))
    if tokens > max_input_tokens:
        raise ValueError(f'Context budget exceeded: {tokens} text tokens > {max_input_tokens}; no history was truncated')
    content = [{'type': 'text', 'text': text}]
    request = {'model': model, 'reasoning_effort': effort,
               'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'run_annotation', 'strict': True, 'schema': SCHEMA}},
               'max_completion_tokens': 16000}
    return request, {'text_input_tokens_estimated': tokens, 'image_count': 0,
                     'image_tokens_in_estimate': False,
                     'packet_sha256': sha(packet), 'request_sha256': sha(request),
                     'rubric_sha256': sha(body['definitions']), 'examples_sha256': sha(examples),
                     'schema_sha256': sha(SCHEMA), 'prompt_sha256': sha(system), 'version': VERSION,
                     'view_version': HTML_INPUT_VERSION, 'html_files_sha256': html_hashes,
                     'view_event_count': len(body['trace']['steps']) + len(body['trace']['planning']),
                     'raw_event_count': len(packet['events'])}


def annotate_one(out, run, client, model='gpt-6-astra', effort='high', max_input_tokens=950000,
                 images=None, execute=True, retries=1):
    out = Path(out)
    rid = run['id']
    if run['status'] not in FINISHED or not run['path']:
        return {'run_id': rid, 'status': 'deferred'}
    lock_path = out / 'locks' / (rid + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        packet = packet_for(run)  # Always fingerprint current evidence, even on resume.
        request, provenance = request_for(out, packet, model, effort, max_input_tokens, images)
        key = provenance['request_sha256']
        result_path = out / 'analyses' / (rid + '.json')
        if result_path.exists():
            previous = load(result_path)['provenance']
            if previous['request_sha256'] == key and previous.get('packet_sha256') == provenance['packet_sha256']:
                return {'run_id': rid, 'status': 'cached'}
        atomic_json(out / 'packets' / (rid + '.json'), packet)
        atomic_json(out / 'requests' / (rid + '.json'), {'request': request, 'provenance': provenance})
        # This is the exact human-readable user payload, with no API envelope or key.
        atomic_json(out / 'model_inputs' / (rid + '.json'), json.loads(request['messages'][1]['content'][0]['text']))
        if not execute:
            return {'run_id': rid, 'status': 'dry_run', **provenance}
        if not json.loads(request['messages'][1]['content'][0]['text'])['trace']['available']:
            return {'run_id': rid, 'status': 'unavailable',
                    'reason': 'No saved HTML action trace; no API call made', **provenance}
        for attempt in range(retries + 1):
            aid = uuid.uuid4().hex
            archive = out / 'attempts' / rid / (aid + '.json')
            record = {'run_id': rid, 'started': now(), 'status': 'requesting', 'request': request, 'provenance': provenance}
            atomic_json(archive, record)
            try:
                response = client.chat.completions.create(**request)
                record.update(response=response.model_dump(mode='json'), finished=now(), status='received')
                atomic_json(archive, record)
                if response.model != model and not response.model.startswith(model + '-'):
                    raise ValueError(f'Returned model {response.model!r} does not match {model!r}')
                choice = response.choices[0]
                if choice.finish_reason != 'stop' or getattr(choice.message, 'refusal', None):
                    raise ValueError('Refused or incomplete model response')
                shown = validation_view(json.loads(request['messages'][1]['content'][0]['text']), packet)
                analysis = validate(json.loads(choice.message.content), shown)
                counts = annotation_counts(analysis, packet['simulator'])
                for family in ['failures', 'inefficiencies']:
                    for x in analysis[family]:
                        x['confidence_calibrated'] = None
                        x['review_status'] = 'unreviewed'
                result = {'run_id': rid, 'analysis': analysis, 'counts': counts, 'simulator': packet['simulator'],
                          'provenance': provenance | {'response_model': response.model, 'attempt_file': str(archive),
                                                     'usage': response.usage.model_dump() if response.usage else {}, 'created': now()}}
                if result_path.exists():
                    old = load(result_path)
                    atomic_json(out / 'history' / rid / (old['provenance']['request_sha256'] + '.json'), old)
                atomic_json(result_path, result)
                record['status'] = 'validated'
                atomic_json(archive, record)
                return {'run_id': rid, 'status': 'annotated'}
            except Exception as exc:
                record.update(status='error', error=str(exc), finished=now())
                atomic_json(archive, record)
                atomic_json(out / 'errors' / (rid + '.json'), {'stage': 'judge', 'error': str(exc), 'attempt_file': str(archive)})
                if attempt == retries:
                    return {'run_id': rid, 'status': 'error', 'error': str(exc)}
                # Retry the exact versioned prompt; never silently change the model/rubric.
                time.sleep(min(10, 2 ** attempt))


def annotation_client():
    """Use the dedicated annotation credential without mutating shared environment."""
    from dotenv import dotenv_values
    from openai import OpenAI
    api_key = os.environ.get('OPENAI_MODEL_PERSONAL')
    if not api_key:
        api_key = dotenv_values(HERE.parents[1] / '.env').get('OPENAI_MODEL_PERSONAL')
    if not api_key:
        raise ValueError('OPENAI_MODEL_PERSONAL is required for annotation calls; no other key is used')
    return OpenAI(api_key=api_key, timeout=300, max_retries=0)


def run_batch(out, workers=2, limit=0, baseline=None, run_id=None, execute=False,
              model='gpt-6-astra', effort='high', max_input_tokens=950000, image_manifest=None):
    check_annotation_scope(baseline, run_id)
    if image_manifest is not None:
        raise ValueError('Astra annotation is text-only; image manifests are not supported')
    out = Path(out)
    manifest = load(out / 'manifest.json')
    # Refresh run completion/path/metrics from source without depending on old catalog state.
    runs = catalog(manifest['source'], manifest.get('overrides'))
    atomic_json(out / 'manifest.json', manifest | {'version': VERSION, 'runs': runs, 'updated': now()})
    chosen = [r for r in runs if r['baseline'] in ANNOTATION_BASELINES
              and (not baseline or r['baseline'] == baseline) and (not run_id or r['id'] == run_id)
              and r['status'] in FINISHED and r['path']]
    if limit:
        chosen = chosen[:limit]
    client = None
    if execute:
        client = annotation_client()
    batch_path = out / 'batches' / (uuid.uuid4().hex + '.json')
    results = []
    last_export = 0.0
    def work(r):
        try:
            return annotate_one(out, r, client, model, effort, max_input_tokens, execute=execute)
        except Exception as exc:
            error = {'run_id': r['id'], 'status': 'error', 'error': str(exc)}
            atomic_json(out / 'errors' / (r['id'] + '.json'), error)
            return error
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, r) for r in chosen]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            atomic_json(batch_path, {'selected': len(chosen), 'finished': len(results), 'execute': execute, 'results': results})
            print(json.dumps(result), flush=True)
            if time.monotonic() - last_export >= 15:
                export(out)
                last_export = time.monotonic()
    export(out)
    return results


def export(out):
    out = Path(out)
    runs = [r for r in load(out / 'manifest.json')['runs'] if r['baseline'] in ANNOTATION_BASELINES]
    (out / 'index.html').write_text((HERE / 'templates/run_judge.html').read_text())
    atomic_json(out / 'input.schema.json', INPUT_SCHEMA)
    atomic_json(out / 'output.schema.json', SCHEMA)
    rows, incidents, review = [], [], []
    prevalence, occurrences, coverage = Counter(), Counter(), Counter()
    for run in runs:
        path = out / 'analyses' / (run['id'] + '.json')
        row = {k: run[k] for k in ['id', 'baseline', 'episode_id', 'repetition', 'status']}
        row['annotation_status'] = 'annotated' if path.exists() else 'not_annotated'
        row['analysis_file'] = path.relative_to(out).as_posix() if path.exists() else None
        if path.exists():
            result = load(path)
            row['simulator_outcome'] = result['simulator']['outcome']
            row['summary'] = result['analysis']['summary']
            row['request_sha256'] = result['provenance']['request_sha256']
            tags, attempts = set(), set()
            for family in ['failures', 'inefficiencies']:
                for x in result['analysis'][family]:
                    incidents.append({'run_id': run['id'], 'family': family, **x})
                    tags.add(x['label'])
                    occurrences[x['label']] += 1
                    if family == 'inefficiencies':
                        attempts.update(x['attempt_ids'])
                    review.append({'run_id': run['id'], 'incident_id': x['id'], 'label': x['label'],
                                   'confidence_raw': x['confidence_raw'], 'request_sha256': row['request_sha256'],
                                   'human_accept': None, 'reviewer': '', 'notes': ''})
            row['labels'] = sorted(tags)
            row['distinct_inefficient_attempts'] = len(attempts)
            prevalence.update(tags)
            for a in result['analysis']['label_assessments']:
                coverage[f'{a["label"]}:{a["status"]}'] += 1
        rows.append(row)
    summary = {'updated': now(), 'annotation_baselines': list(ANNOTATION_BASELINES),
               'total_runs': len(rows), 'annotated': sum(r['annotation_status'] == 'annotated' for r in rows),
               'label_run_counts': dict(prevalence), 'label_incident_counts': dict(occurrences),
               'assessment_coverage': dict(coverage), 'runs': rows,
               'note': 'Model annotations are unreviewed. Counts overlap; unavailable scores and unassessed labels are not negatives.'}
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / 'incidents.json', incidents)
    atomic_json(out / 'review_template.json', review)
    for name, data in [('runs.csv', rows), ('incidents.csv', incidents)]:
        stream = io.StringIO()
        keys = sorted({k for row in data for k in row})
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        temporary = out / (name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(stream.getvalue())
        temporary.replace(out / name)
    return summary


def make_example(out, run_id, name):
    """Create a disabled draft for human correction, never auto-approve AI labels."""
    import re
    out = Path(out)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Example name must contain only letters, digits, underscores or hyphens')
    if run_id not in {r['id'] for r in load(out / 'manifest.json')['runs']}:
        raise ValueError('Unknown run ID')
    destination = out / 'examples' / (name + '.json')
    if destination.exists():
        raise ValueError('Example already exists; edit it rather than overwriting it')
    result = load(out / 'analyses' / (run_id + '.json'))
    packet = load(out / 'packets' / (run_id + '.json'))
    extra = {}
    if result.get('provenance', {}).get('view_version', '').startswith('html-judge-'):
        # The current draft may have changed after this paid annotation. Recover
        # the exact request that produced its step IDs, not the latest preview.
        attempt = load(Path(result['provenance']['attempt_file']))
        extra['model_input'] = json.loads(attempt['request']['messages'][1]['content'][0]['text'])
    atomic_json(destination, {
        'enabled': False, 'reviewed_by': '', 'source_run_ids': [run_id],
        'instructions': 'AI-generated draft. Correct expected_output, enter your reviewer name, then enable. Never use this source run in held-out evaluation.',
        'input': packet, 'expected_output': result['analysis'],
        **extra,
    })
    return {'example': str(destination), 'enabled': False}


def calibration_report(out, reviews):
    """Audit stated confidence against independently reviewed incident correctness.

    This measures precision/calibration of emitted incidents, NOT missed-incident
    recall and NOT a fitted calibration model. No probabilities are fabricated.
    """
    out = Path(out)
    groups = defaultdict(list)
    seen = set()
    for row in load(reviews):
        if row.get('human_accept') is None:
            continue
        if type(row['human_accept']) is not bool or not row.get('reviewer'):
            raise ValueError('Human review requires boolean human_accept and reviewer')
        key = (row['run_id'], row['incident_id'])
        if key in seen:
            raise ValueError('Duplicate human review')
        seen.add(key)
        result = load(out / 'analyses' / (row['run_id'] + '.json'))
        if result['provenance']['request_sha256'] != row['request_sha256']:
            raise ValueError('Review targets a different annotation version')
        incident = next(x for family in ['failures', 'inefficiencies'] for x in result['analysis'][family] if x['id'] == row['incident_id'])
        pair = (incident['confidence_raw'], int(row['human_accept']))
        groups['all'].append(pair)
        groups[incident['label']].append(pair)
    report = {}
    for label, values in groups.items():
        bins = []
        for i in range(10):
            b = [(p, y) for p, y in values if min(9, int(p * 10)) == i]
            if b:
                bins.append({'lower': i / 10, 'upper': (i + 1) / 10, 'n': len(b),
                             'mean_confidence': sum(p for p, _ in b) / len(b), 'observed_accuracy': sum(y for _, y in b) / len(b)})
        report[label] = {'n': len(values), 'accepted_fraction': sum(y for _, y in values) / len(values),
                         'brier': sum((p - y) ** 2 for p, y in values) / len(values),
                         'ece': sum(b['n'] * abs(b['mean_confidence'] - b['observed_accuracy']) for b in bins) / len(values),
                         'reliability_bins': bins}
    result = {'metrics': report, 'note': 'Incident correctness calibration only; representative independent reviews required. Recall requires separately annotated missed incidents. No calibration mapping fitted.'}
    atomic_json(out / 'calibration_report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    sub = parser.add_subparsers(dest='cmd', required=True)
    prep = sub.add_parser('prepare', help='Import runs and build evidence without API calls')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--overrides', type=Path, help='Replacement-run manifest, persisted for subsequent batches')
    run = sub.add_parser('run', help='Build requests; add --execute to send them')
    for p in [prep, run]:
        p.add_argument('--limit', type=int, default=0)
        p.add_argument('--baseline', choices=ANNOTATION_BASELINES)
        p.add_argument('--run-id')
    run.add_argument('--execute', action='store_true')
    run.add_argument('--workers', type=int, default=2)
    run.add_argument('--model', default='gpt-6-astra')
    run.add_argument('--effort', default='high', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    run.add_argument('--max-input-tokens', type=int, default=950000)
    sub.add_parser('export')
    sub.add_parser('status')
    ex = sub.add_parser('example', help='Create a disabled prompt-example draft from an annotation')
    ex.add_argument('--run-id', required=True)
    ex.add_argument('--name', required=True)
    cal = sub.add_parser('calibration')
    cal.add_argument('--reviews', type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, 'limit', 0) < 0 or getattr(args, 'workers', 1) < 1:
        parser.error('Limit must be nonnegative and workers must be positive')
    if args.cmd == 'prepare':
        result = prepare(args.source, args.out, args.limit, args.baseline, args.run_id, args.overrides)
    elif args.cmd == 'run':
        result = run_batch(args.out, args.workers, args.limit, args.baseline, args.run_id,
                           args.execute, args.model, args.effort, args.max_input_tokens)
    elif args.cmd == 'example':
        result = make_example(args.out, args.run_id, args.name)
    elif args.cmd == 'calibration':
        result = calibration_report(args.out, args.reviews)
    else:
        result = export(args.out)
        result = {k: v for k, v in result.items() if k != 'runs'}
    print(json.dumps(result, indent=2))
    if args.cmd == 'run' and any(r.get('status') == 'error' for r in result):
        raise SystemExit(1)
    if args.cmd == 'prepare' and result.get('error'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
