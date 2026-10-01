"""Saved criterion results and action attribution, independent of Habitat."""
import copy
import unittest
from scripts.ground_truth_evidence import criteria_evidence


def fixture():
    return {
        'metadata': {'success_criteria': '- is_filled(jug_0)\n- is_on_top(jug_0, table_0)',
                     'propositions': [
                         {'function_name': 'is_filled', 'args': {'object_handles': ['jug'], 'number': 1}},
                         {'function_name': 'is_on_top', 'args': {'object_handles': ['jug'], 'receptacle_handles': ['table'], 'number': 1}}]},
        'snapshot': {'entities': {'objects': [{'name': 'jug_1', 'sim_handle': 'jug'}],
                                  'furniture': [{'name': 'table_9', 'sim_handle': 'table'}]},
                     'aliases': {'jug_0': 'jug_1', 'table_0': 'table_9'},
                     'evaluation': {'success': True, 'satisfied_at': [2, 3], 'current': [True, True],
                                    'constraint_satisfaction': [[True, True]]}},
        'actions': [{'sequence': 1, 'skill': 'Fill', 'target': 'jug_0',
                     'result': {'ok': True, 'skill_steps': 2, 'recorded_frames': 2}},
                    {'sequence': 2, 'skill': 'Navigate', 'target': 'table_0',
                     'result': {'ok': True, 'skill_steps': 3, 'recorded_frames': 3}}]}


class CriteriaEvidenceTests(unittest.TestCase):
    def test_first_satisfaction_maps_exact_boundaries_and_does_not_guess_causality(self):
        run = fixture()
        original = copy.deepcopy(run)
        result = criteria_evidence(run)
        self.assertEqual(result['status'], 'pass')
        rows = result['criteria']
        self.assertEqual([row['evidence'][0]['sequence'] for row in rows], [1, 2])
        self.assertEqual(rows[1]['evidence'][0]['skill'], 'Navigate')
        self.assertAlmostEqual(rows[0]['evidence'][0]['video_time'], 1 / 30)
        self.assertAlmostEqual(rows[1]['evidence'][0]['video_time'], 2 / 30)
        self.assertEqual(run, original)

    def test_evidence_uses_real_video_presentation_times_for_legacy_clips(self):
        run = fixture()
        row = criteria_evidence(run, frame_times=[0, .034, .071, .104, .138])['criteria'][1]
        self.assertEqual(row['evidence'][0]['video_time'], .071)
        row = criteria_evidence(run, frame_times=[])['criteria'][1]
        self.assertIsNone(row['evidence'][0]['video_time'])
        self.assertEqual(row['evidence'][0]['sequence'], 2)

    def test_initial_state_needs_no_action(self):
        run = fixture()
        run['snapshot']['evaluation']['satisfied_at'][0] = 0
        row = criteria_evidence(run)['criteria'][0]
        self.assertEqual(row['status'], 'pass')
        self.assertEqual(row['evidence'], [])
        self.assertIn('initial scene', row['detail'])

    def test_successful_skills_do_not_prove_task_success(self):
        run = fixture()
        run['snapshot']['evaluation'].update(success=False, satisfied_at=[-1, 3], current=[False, True])
        result = criteria_evidence(run)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['criteria'][0]['evidence'], [])
        self.assertEqual(result['met'], 1)

    def test_final_state_and_constraint_failures_are_visible(self):
        for field, value in [('current', [False, True]), ('constraint_satisfaction', [[False, True]])]:
            with self.subTest(field=field):
                run = fixture()
                run['snapshot']['evaluation'][field] = value
                result = criteria_evidence(run)
                self.assertEqual(result['status'], 'fail')
                self.assertEqual(result['criteria'][0]['evidence'][0]['sequence'], 1)

    def test_order_and_transient_requirements_include_both_steps(self):
        run = fixture()
        run['metadata']['success_criteria'] += '\n- order: is_filled(jug_0) before is_on_top(jug_0, table_0)'
        run['snapshot']['evaluation']['current'][0] = False
        result = criteria_evidence(run)
        self.assertEqual(result['status'], 'pass')
        self.assertIn('intermediate', result['criteria'][0]['detail'])
        self.assertEqual([e['sequence'] for e in result['criteria'][2]['evidence']], [1, 2])
        run['snapshot']['evaluation']['satisfied_at'] = [4, 3]
        self.assertEqual(criteria_evidence(run)['criteria'][2]['status'], 'fail')

    def test_report_requirements_and_zero_step_timing(self):
        run = fixture()
        report = 'Robot should report that no suitable object exists.'
        run['metadata']['success_criteria'] += '\n- ' + report
        run['metadata']['reports'] = ['- ' + report]
        self.assertEqual(criteria_evidence(run)['status'], 'fail')
        run['actions'].insert(0, {'sequence': 0, 'skill': 'ReportAbsence', 'target': '- ' + report,
                                  'result': {'ok': True, 'skill_steps': 0}})
        result = criteria_evidence(run)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['criteria'][-1]['evidence'][0]['skill'], 'ReportAbsence')
        self.assertAlmostEqual(result['criteria'][1]['evidence'][0]['video_time'], 2 / 30)

    def test_partial_timing_does_not_misattribute_later_success(self):
        run = fixture()
        del run['actions'][0]['result']['skill_steps']
        row = criteria_evidence(run)['criteria'][1]
        self.assertEqual(row['status'], 'pass')
        self.assertEqual(row['evidence'], [])
        self.assertIn('incomplete action timing', row['detail'])
        run = fixture()
        run['actions'][0]['result'].update(ok=False, skill_steps=0)
        self.assertEqual(criteria_evidence(run)['criteria'][1]['evidence'], [])

    def test_missing_evaluator_unknown_text_and_stale_recording(self):
        run = fixture()
        run['snapshot']['evaluation'] = {}
        self.assertEqual(criteria_evidence(run)['status'], 'unknown')
        run = fixture()
        run['metadata']['success_criteria'] += '\n- Inspect whether the table looks tidy.'
        self.assertEqual(criteria_evidence(run)['status'], 'unknown')
        run = fixture()
        run['stale'] = True
        self.assertEqual(criteria_evidence(run)['status'], 'outdated')
        self.assertEqual(criteria_evidence(run)['recorded_status'], 'pass')

    def test_distractor_identity_guard_uses_actual_predicate_results(self):
        run = fixture()
        run['metadata']['success_criteria'] += '\n- Distractors (jug_1) must not be used in place of task objects.'
        row = criteria_evidence(run)['criteria'][-1]
        self.assertEqual(row['status'], 'pass')
        self.assertEqual(len(row['evidence']), 2)
        self.assertIn('no separate completion action', row['detail'])
        run['snapshot']['evaluation']['satisfied_at'][1] = -1
        self.assertEqual(criteria_evidence(run)['criteria'][-1]['status'], 'fail')

    def test_unrolled_or_missing_predicate_data_is_not_claimed_verified(self):
        run = fixture()
        run['snapshot']['evaluation']['satisfied_at'].append(4)
        result = criteria_evidence(run)
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(all(not row['evidence'] for row in result['criteria']))


if __name__ == '__main__':
    unittest.main()
