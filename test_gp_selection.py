import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import gp_transfer as gp


class PrimitiveContractTests(unittest.TestCase):
    def test_environment_availability_and_values(self):
        cartpole = gp.summaries([0.6, 0.0, -0.8, 0.0], 1, 2, 2, 'cartpole')
        ttt_state = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 9
        ttt = gp.summaries(ttt_state, 1, 9, 5, 'ttt')

        self.assertIn('state_magnitude', gp.available_primitives('ttt'))
        self.assertIn('state_dispersion', gp.available_primitives('cartpole'))
        self.assertNotIn('state_boundary_margin', gp.available_primitives('ttt'))
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_magnitude')], 0.5)
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_boundary_margin')], 0.2)
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_min')], -0.8)
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_range')], 1.4)
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_positive_fraction')], 0.25)
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_negative_fraction')], 0.25)
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_sparsity')], 0.5)
        self.assertTrue(np.isnan(ttt[gp.PRIMS.index('state_boundary_margin')]))
        self.assertIn('transition_delta_mean', gp.available_primitives('cartpole'))
        self.assertIn('transition_delta_mean', gp.available_primitives('lineworld'))
        self.assertNotIn('transition_delta_mean', gp.available_primitives('ttt'))
        temporal = gp.summaries([0.2, 0.4], 1, 2, 2, 'lineworld', [0.1, -0.1])
        self.assertAlmostEqual(temporal[gp.PRIMS.index('transition_delta_mean')], 0.0)
        self.assertAlmostEqual(temporal[gp.PRIMS.index('transition_delta_abs_mean')], 0.1)
        self.assertAlmostEqual(temporal[gp.PRIMS.index('transition_delta_max_abs')], 0.1)
        self.assertAlmostEqual(temporal[gp.PRIMS.index('transition_state_alignment')], -1.0 / np.sqrt(10.0))
        self.assertIn('board_own_immediate_wins', gp.available_primitives('ttt'))
        self.assertIn('board_own_immediate_wins', gp.available_primitives('connect_four'))
        self.assertNotIn('board_own_immediate_wins', gp.available_primitives('cartpole'))
        board_features = gp.board_structural_features(ttt_state, 'ttt')
        self.assertAlmostEqual(board_features['board_own_immediate_wins'], 1.0 / 9.0)
        self.assertGreater(board_features['board_own_open_lines'], 0.0)
        self.assertAlmostEqual(board_features['board_occupancy'], 2.0 / 9.0)
        c4_state = [0.0] * 84
        for index in (35, 36, 37): c4_state[index] = 1.0
        c4_features = gp.board_structural_features(c4_state, 'connect_four')
        self.assertAlmostEqual(c4_features['board_own_immediate_wins'], 1.0 / 42.0)
        self.assertAlmostEqual(c4_features['board_occupancy'], 3.0 / 42.0)
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_median')], 0.0)
        self.assertAlmostEqual(cartpole[gp.PRIMS.index('state_iqr')], 0.35)
        self.assertTrue(np.isfinite(cartpole[gp.PRIMS.index('state_skewness')]))
        self.assertGreaterEqual(cartpole[gp.PRIMS.index('state_entropy')], 0.0)
        self.assertLessEqual(cartpole[gp.PRIMS.index('state_entropy')], 1.0)
        self.assertIn('state_position_moment', gp.available_primitives('ttt'))
        self.assertAlmostEqual(
            gp.summaries([1.0, 0.0, -1.0], 1, 3, 3, 'nim')[gp.PRIMS.index('state_adjacent_difference')],
            1.0,
        )
        self.assertAlmostEqual(
            gp.summaries([1.0, -1.0, 1.0], 1, 3, 3, 'nim')[gp.PRIMS.index('state_sign_change_fraction')],
            1.0,
        )

    def test_tree_generation_respects_environment_primitives(self):
        python_rng = random.Random(42)
        allowed = gp.available_primitives('ttt')
        trees = [gp.randtree(python_rng, primitives=allowed) for _ in range(100)]

        for tree in trees:
            self.assertTrue(set(gp.dependencies(tree)).issubset(allowed))

    def test_generator_uses_advanced_functions_and_evaluates_them(self):
        python_rng = random.Random(42)
        trees = [gp.randtree(python_rng) for _ in range(300)]
        unary = {
            node[0]
            for tree in trees
            for _, node in gp.walk_paths(tree)
            if not isinstance(node, str) and gp.ARITY[node[0]] == 1
        }

        self.assertEqual(unary, set(gp.UNARY_OPS))
        features = np.zeros((len(gp.PRIMS), 1))
        features[gp.PRIMS.index('state_min'), 0] = -0.5
        value = gp.ev(('sqrt', ('neg', 'state_min')), features)[0]
        self.assertAlmostEqual(value, np.tanh(np.sqrt(0.5)))

    def test_new_game_datasets_match_shared_primitive_contract(self):
        for game in ('ttt', 'cartpole', 'connect_four', 'nim', 'lineworld'):
            with self.subTest(game=game):
                features, advantages = gp.make_dataset(game, 23, nstates=8, rollouts=1)
                self.assertEqual(features.shape[0], len(gp.PRIMS))
                self.assertEqual(features.shape[1], len(advantages))
                self.assertGreater(features.shape[1], 0)
                available_rows = [gp.PRIMS.index(name) for name in gp.available_primitives(game)]
                unavailable_rows = sorted(set(range(len(gp.PRIMS))) - set(available_rows))
                self.assertTrue(np.all(np.isfinite(features[available_rows])))
                self.assertTrue(np.all(np.isnan(features[unavailable_rows])))
                self.assertTrue(np.all(np.isfinite(advantages)))

    def test_connect_four_win_detection_in_all_directions(self):
        cases = [
            [(5, 0), (5, 1), (5, 2), (5, 3)],
            [(2, 0), (3, 0), (4, 0), (5, 0)],
            [(2, 0), (3, 1), (4, 2), (5, 3)],
            [(2, 6), (3, 5), (4, 4), (5, 3)],
        ]
        for cells in cases:
            board = [[0] * 7 for _ in range(6)]
            for row, col in cells:
                board[row][col] = 1
            self.assertTrue(gp.connect_four_wins(board, 1))


class MultiObjectiveTests(unittest.TestCase):
    def test_nondominated_sort_preserves_tradeoffs(self):
        objectives = [
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, -1.0, 1.0),
            (0.5, 0.5, -2.0, 0.1),
            (0.0, 0.0, -1.0, 0.0),
        ]

        fronts = gp.nondominated_fronts(objectives)

        self.assertEqual(set(fronts[0]), {0, 1, 2})
        self.assertEqual(fronts[1], [3])
        order, ranks, crowding = gp.nsga_ranking(objectives)
        self.assertEqual(set(order[:3]), {0, 1, 2})
        self.assertEqual(ranks[3], 1)
        self.assertTrue(all(crowding[index] >= 0 for index in range(4)))

    def test_signed_fitness_prefers_correct_direction_and_generalization(self):
        X = np.zeros((len(gp.PRIMS), 12))
        signal = np.linspace(-1.0, 1.0, 12)
        X[gp.PRIMS.index('state_mean')] = signal
        y = signal.copy()
        good = gp.population_objectives(['state_mean'], X, y, X, y)[0]
        inverted = gp.population_objectives([('neg', 'state_mean')], X, y, X, y)[0]
        self.assertGreater(good[0], inverted[0])
        self.assertGreater(good[0], 0.99)
        self.assertAlmostEqual(good[2], 0.0)

    def test_overfit_candidate_is_penalized_by_train_validation_gap(self):
        X = np.zeros((len(gp.PRIMS), 12))
        Xval = np.zeros_like(X)
        signal = np.linspace(-1.0, 1.0, 12)
        X[gp.PRIMS.index('state_mean')] = signal
        Xval[gp.PRIMS.index('state_mean')] = signal[::-1]
        y = signal.copy()
        yval = signal.copy()
        score = gp.population_objectives(['state_mean'], X, y, Xval, yval)[0]
        self.assertLess(score[0], 0.0)
        self.assertLess(score[2], 0.0)

    def test_constants_are_screened_below_predictive_candidates(self):
        train = np.zeros((len(gp.PRIMS), 8))
        train[gp.PRIMS.index('action_norm')] = np.tile([-1.0, 1.0], 4)
        train[gp.PRIMS.index('state_mean')] = np.linspace(-1.0, 1.0, 8)
        labels = train[gp.PRIMS.index('action_norm')].copy()

        scores = gp.population_objectives(
            ['action_norm', ('div', 'action_norm', 'action_norm')],
            train, labels, train, labels,
        )

        self.assertGreater(scores[0][0], scores[1][0])
        self.assertGreater(scores[0][1], scores[1][1])


class StagnationRestartTests(unittest.TestCase):
    class StagnantMonitor:
        def __init__(self):
            self.window = 2
            self.stagnant = 0
            self.struct = set()
            self.behaviors = []

        def observe(self, population):
            self.stagnant += 1
            return {
                'structurally_new': len(population),
                'behaviorally_new': 0,
                'behavioral_diversity': 0.0,
            }

        def reset_stagnation(self):
            self.stagnant = 0

    class NovelAfterRestartMonitor(StagnantMonitor):
        def __init__(self):
            super().__init__()
            self.just_restarted = False

        def observe(self, population):
            if self.just_restarted:
                self.just_restarted = False
                self.stagnant = 0
                return {
                    'structurally_new': len(population),
                    'behaviorally_new': 1,
                    'behavioral_diversity': 0.5,
                }
            return super().observe(population)

        def reset_stagnation(self):
            super().reset_stagnation()
            self.just_restarted = True

    def test_restart_preserves_elite_and_injects_fresh_expressions(self):
        elite = 'action_norm'
        population = gp.restart_population(
            [elite], 8, random.Random(7), gp.available_primitives('nim'),
            random_fraction=.75,
        )

        self.assertEqual(len(population), 8)
        self.assertEqual(population[0], elite)
        self.assertTrue(all(set(gp.dependencies(tree)).issubset(gp.available_primitives('nim'))
                            for tree in population))
        self.assertGreater(len({gp.canon(tree) for tree in population}), 1)

    def test_evolution_restarts_and_stops_after_configured_failures(self):
        X = np.zeros((len(gp.PRIMS), 8))
        X[gp.PRIMS.index('action_norm')] = np.tile([-1.0, 1.0], 4)
        X[gp.PRIMS.index('state_mean')] = np.linspace(-1.0, 1.0, 8)
        y = X[gp.PRIMS.index('action_norm')].copy()
        population = ['action_norm', 'state_mean', 'state_abs_mean', 'state_var']
        monitor = self.StagnantMonitor()

        final_population, archive, history, restarts = gp.evolve_population(
            population, X, y, X, y, monitor, random.Random(17),
            gp.available_primitives('nim'), generations=10,
            stagnation_window=2, max_unsuccessful_restarts=2,
            restart_random_fraction=.5, restart_mutation_generations=3,
        )

        self.assertEqual(len(final_population), len(population))
        self.assertTrue(archive)
        self.assertTrue(restarts['stopped_after_unsuccessful_restarts'])
        self.assertEqual(restarts['unsuccessful_restarts'], 2)
        self.assertEqual(restarts['restarts_triggered'], 2)
        triggered = [entry for entry in history if entry['restart_triggered']]
        self.assertEqual(len(triggered), 2)
        self.assertTrue(all(entry['preserved_elites'] > 0 for entry in triggered))
        self.assertTrue(all(entry['random_individuals_injected'] > 0 for entry in triggered))
        self.assertTrue(all(entry['high_mutation_generations'] == 3 for entry in triggered))

    def test_restart_success_is_recorded_when_new_behavior_appears(self):
        X = np.zeros((len(gp.PRIMS), 8))
        X[gp.PRIMS.index('action_norm')] = np.tile([-1.0, 1.0], 4)
        X[gp.PRIMS.index('state_mean')] = np.linspace(-1.0, 1.0, 8)
        y = X[gp.PRIMS.index('action_norm')].copy()
        population = ['action_norm', 'state_mean', 'state_abs_mean', 'state_var']

        _, _, history, restarts = gp.evolve_population(
            population, X, y, X, y, self.NovelAfterRestartMonitor(),
            random.Random(19), gp.available_primitives('nim'), generations=4,
            stagnation_window=2, max_unsuccessful_restarts=2,
        )

        self.assertEqual(restarts['successful_restarts'], 1)
        self.assertEqual(restarts['unsuccessful_restarts'], 0)
        self.assertEqual(restarts['events'][0]['outcome'], 'successful')
        self.assertEqual(restarts['events'][0]['new_behaviors'], 1)
        self.assertTrue(any(entry['restart_outcome'] == 'successful' for entry in history))


class DownstreamAdmissionTests(unittest.TestCase):
    def test_gate_compares_same_shape_feature_sets_on_paired_seeds(self):
        candidate = {'expression': 'state_magnitude'}

        def fake_trial(_brain, _target, expressions, _budget, seed, _population):
            self.assertEqual(len(expressions), 5)
            self.assertEqual(expressions[0], 'action_norm')
            if expressions[1] == 'sub(state_mean,state_mean)':
                return 10.0
            return {11: 12.0, 22: 11.0, 33: 9.0}[seed]

        with patch.object(gp, 'resolve_brain_path', return_value=Path('brain')), \
                patch.object(gp, 'validate_brain_contract'), \
                patch.object(gp, 'run_brain_trial', side_effect=fake_trial):
            gp.evaluate_downstream(
                [candidate], 'cartpole', 1000, [11, 22, 33], 12, 'brain',
            )

        utility = candidate['downstream_utility']
        self.assertEqual(utility['mean_paired_difference'], 2.0 / 3.0)
        self.assertAlmostEqual(utility['win_fraction'], 2.0 / 3.0)
        self.assertTrue(utility['passed'])
        self.assertEqual(candidate['expression'], 'state_magnitude')

    def test_gate_rejects_one_lucky_seed(self):
        candidate = {'expression': 'state_magnitude'}

        def fake_trial(_brain, _target, expressions, _budget, seed, _population):
            if expressions[1] == 'sub(state_mean,state_mean)':
                return 10.0
            return {1: 20.0, 2: 9.0, 3: 9.0, 4: 9.0, 5: 9.0}[seed]

        with patch.object(gp, 'resolve_brain_path', return_value=Path('brain')), \
                patch.object(gp, 'validate_brain_contract'), \
                patch.object(gp, 'run_brain_trial', side_effect=fake_trial):
            gp.evaluate_downstream([candidate], 'cartpole', 1000, [1, 2, 3, 4, 5], 8, 'brain')

        utility = candidate['downstream_utility']
        self.assertGreater(utility['mean_paired_difference'], 0.0)
        self.assertEqual(utility['win_fraction'], 0.2)
        self.assertFalse(utility['passed'])


if __name__ == '__main__':
    unittest.main()


class CounterfactualDataTests(unittest.TestCase):
    def test_common_random_numbers_pair_rollout_seeds_across_actions(self):
        seen = {0: [], 1: []}

        def rollout(action, rollout_rng):
            noise = rollout_rng.random()
            seen[action].append(noise)
            return noise + 0.25 * action

        means, samples = gp.paired_action_values(
            [0, 1], rollout, rollouts=20, rng=random.Random(1234)
        )
        self.assertEqual(seen[0], seen[1])
        np.testing.assert_allclose(samples[0], np.asarray(samples[1]) - 0.25, atol=1e-15)
        self.assertAlmostEqual(means[1] - means[0], 0.25, places=12)

    def test_paired_action_values_rejects_zero_rollouts(self):
        with self.assertRaises(ValueError):
            gp.paired_action_values([0, 1], lambda action, rng: 0.0, 0, random.Random(1))

    def test_dataset_generation_is_reproducible_with_paired_rollouts(self):
        first_x, first_y = gp.make_dataset('nim', 5150, nstates=12, rollouts=8)
        second_x, second_y = gp.make_dataset('nim', 5150, nstates=12, rollouts=8)
        np.testing.assert_allclose(first_x, second_x, equal_nan=True)
        np.testing.assert_allclose(first_y, second_y)

    def test_independent_mode_uses_distinct_seed_sequences(self):
        seen = {0: [], 1: []}

        def rollout(action, rollout_rng):
            value = rollout_rng.random()
            seen[action].append(value)
            return value

        gp.paired_action_values(
            [0, 1], rollout, rollouts=20, rng=random.Random(4321),
            common_random_numbers=False,
        )
        self.assertNotEqual(seen[0], seen[1])


class OracleTeacherTests(unittest.TestCase):
    def test_tictactoe_oracle_finds_immediate_win(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        actions, values = gp.ttt_oracle_action_values(board)
        self.assertEqual(actions[values.argmax()], 2)
        self.assertEqual(values.max(), 1.0)

    def test_nim_oracle_identifies_losing_and_winning_positions(self):
        actions4, values4 = gp.nim_oracle_action_values(4)
        self.assertEqual(actions4, [0, 1, 2])
        self.assertTrue(np.all(values4 == -1.0))
        actions5, values5 = gp.nim_oracle_action_values(5)
        self.assertEqual(values5[0], 1.0)
        self.assertTrue(np.all(values5[1:] == -1.0))

    def test_lineworld_oracle_prefers_progress_toward_goal(self):
        actions, values = gp.lineworld_oracle_action_values(4, 5)
        self.assertEqual(actions[values.argmax()], 1)

    def test_oracle_datasets_are_finite_and_repeatable(self):
        for game in ('ttt', 'nim', 'lineworld'):
            with self.subTest(game=game):
                x1, y1 = gp.make_dataset(game, 712, nstates=12, teacher_policy='oracle')
                x2, y2 = gp.make_dataset(game, 712, nstates=12, teacher_policy='oracle')
                self.assertTrue(np.isfinite(y1).all())
                np.testing.assert_allclose(x1, x2, equal_nan=True)
                np.testing.assert_allclose(y1, y2)

    def test_oracle_mode_rejects_games_without_exact_teacher(self):
        with self.assertRaises(ValueError):
            gp.make_dataset('cartpole', 1, nstates=2, teacher_policy='oracle')
