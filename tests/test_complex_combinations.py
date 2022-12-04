import bt4vt
import filecmp


class TestComplexCombinations:
    def test_subgroup_not_in_metadata(self):
        # Test Case 1a: Subgroup not available in speaker metadata
        config_1a = "./tests/complex_tests/config_1a.yaml"
        scores_1a = "./tests/complex_tests/scores_1a.csv"

        test_1a = bt4vt.core.SpeakerBiasTest(scores_1a, config_1a)
        test_1a.run_tests()

        assert filecmp.cmp("./tests/complex_tests/results/biastest_results_config_1a_scores_1a.csv",
                           "./tests/complex_tests/reference_results/reference_biastest_results_config_1a_scores_1a.csv",
                           shallow=False) == True

    def test_subgroup_combination_not_in_metadata(self):
        #Test Case 1b: Subgroup combination not available in speaker metadata
        config_1b = "./tests/complex_tests/config_1b.yaml"
        scores_1b = "./tests/complex_tests/scores_1b.csv"

        test_1b = bt4vt.core.SpeakerBiasTest(scores_1b, config_1b)
        test_1b.run_tests()

        assert filecmp.cmp("./tests/complex_tests/results/biastest_results_config_1b_scores_1b.csv",
                           "./tests/complex_tests/reference_results/reference_biastest_results_config_1b_scores_1b.csv",
                           shallow=False) == True

    def test_one_category_with_scores(self):
        # Test Case 2: Only one category of subgroup with scores (current libri test) - female data available in metadata but no scores
        config_2 = "./tests/complex_tests/config_2.yaml"
        scores_2 = "./tests/complex_tests/scores_2.csv"

        test_2 = bt4vt.core.SpeakerBiasTest(scores_2, config_2)
        test_2.run_tests()

        assert filecmp.cmp("./tests/complex_tests/results/biastest_results_config_2_scores_2.csv",
                           "./tests/complex_tests/reference_results/reference_biastest_results_config_2_scores_2.csv",
                           shallow=False) == True

    def test_additional_metadata(self):
        # Test Case 3: “Any” vs. “All” check - additional male score in metadata but not in scores
        config_3 = "./tests/complex_tests/config_3.yaml"
        scores_3 = "./tests/complex_tests/scores_3.csv"

        test_3 = bt4vt.core.SpeakerBiasTest(scores_3, config_3)
        test_3.run_tests()

        assert filecmp.cmp("./tests/complex_tests/results/biastest_results_config_3_scores_3.csv",
                            "./tests/complex_tests/reference_results/reference_biastest_results_config_3_scores_3.csv",
                            shallow=False) == True
