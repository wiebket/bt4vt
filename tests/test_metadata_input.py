import bt4vt
import filecmp


class TestMetadataInput:

    def test_additional_column(self):
        # Test Case 1: Additional Columns we are not using
        config_1 = './tests/metadata_tests/config_1.yaml'
        scores_1 = './tests/metadata_tests/scores_1.csv'

        test_1 = bt4vt.core.SpeakerBiasTest(scores_1, config_1)
        test_1.run_tests()

        assert filecmp.cmp("./tests/metadata_tests/results/biastest_results_config_1_scores_1.csv",
                           "./tests/metadata_tests/reference_results/reference_biastest_results_config_1_scores_1.csv",
                           shallow=False) == True

    def test_different_separator(self):
        # Test Case 2: Separator ";" in metadata file
        # TODO more separators might need to be checked
        config_2 = "./tests/metadata_tests/config_2.yaml"
        scores_2 = "./tests/metadata_tests/scores_2.csv"

        test_2 = bt4vt.core.SpeakerBiasTest(scores_2, config_2)
        test_2.run_tests()

        assert filecmp.cmp("./tests/metadata_tests/results/biastest_results_config_2_scores_2.csv",
                           "./tests/metadata_tests/reference_results/reference_biastest_results_config_2_scores_2.csv",
                           shallow=False) == True

    def test_additional_ids(self):
        # Test Case 3: More people in metadata than scores
        config_3 = "./tests/metadata_tests/config_3.yaml"
        scores_3 = "./tests/metadata_tests/scores_3.csv"

        test_3 = bt4vt.core.SpeakerBiasTest(scores_3, config_3)
        test_3.run_tests()

        assert filecmp.cmp("./tests/metadata_tests/results/biastest_results_config_3_scores_3.csv",
                           "./tests/metadata_tests/reference_results/reference_biastest_results_config_3_scores_3.csv",
                           shallow=False) == True

    def test_missing_ids(self):
        # Test Case 4: Missing IDs, ids in scores file that are not in metadata file
        config_4 = "./tests/metadata_tests/config_4.yaml"
        scores_4 = "./tests/metadata_tests/scores_4.csv"

        test_4 = bt4vt.core.SpeakerBiasTest(scores_4, config_4)
        test_4.run_tests()

        assert filecmp.cmp("./tests/metadata_tests/results/biastest_results_config_4_scores_4.csv",
                           "./tests/metadata_tests/reference_results/reference_biastest_results_config_4_scores_4.csv",
                           shallow=False) == True

    def test_one_category(self):
        # Test Case 5: Only one category of subgroup, e.g. only females
        config_5 = "./tests/metadata_tests/config_5.yaml"
        scores_5 = "./tests/metadata_tests/scores_5.csv"

        test_5 = bt4vt.core.SpeakerBiasTest(scores_5, config_5)
        test_5.run_tests()

        assert filecmp.cmp("./tests/metadata_tests/results/biastest_results_config_5_scores_5.csv",
                           "./tests/metadata_tests/reference_results/reference_biastest_results_config_5_scores_5.csv",
                           shallow=False) == True

    def test_none_label(self):
        # Test Case 6: None group label
        # This is probably not what we want to test as the subgroup label is simply None now
        config_6 = "./tests/metadata_tests/config_6.yaml"
        scores_6 = "./tests/metadata_tests/scores_6.csv"

        test_6 = bt4vt.core.SpeakerBiasTest(scores_6, config_6)
        test_6.run_tests()

        assert filecmp.cmp("./tests/metadata_tests/results/biastest_results_config_6_scores_6.csv",
                           "./tests/metadata_tests/reference_results/reference_biastest_results_config_6_scores_6.csv",
                           shallow=False) == True

    def test_empty_label(self):
        # Test Case 7: Empty Group Label
        config_7 = "./tests/metadata_tests/config_7.yaml"
        scores_7 = "./tests/metadata_tests/scores_7.csv"

        test_7 = bt4vt.core.SpeakerBiasTest(scores_7, config_7)
        test_7.run_tests()

        assert filecmp.cmp("./tests/metadata_tests/results/biastest_results_config_7_scores_7.csv",
                           "./tests/metadata_tests/reference_results/reference_biastest_results_config_7_scores_7.csv",
                           shallow=False) == True
