import bt4vt
import filecmp
import pytest


class TestScoresInput:
    # Test Case 1: Different Column Headers - ref file
    def test_different_ref_column(self):
        config_1a = "./tests/scoresfile_tests/config_1a.yaml"
        scores_1a = "./tests/scoresfile_tests/scores_1a.csv"

        pytest.raises(SystemExit, bt4vt.core.SpeakerBiasTest, scores_1a, config_1a)

    def test_different_com_column(self):
        config_1b = "./tests/scoresfile_tests/config_1b.yaml"
        scores_1b = "./tests/scoresfile_tests/scores_1b.csv"

        pytest.raises(SystemExit, bt4vt.core.SpeakerBiasTest, scores_1b, config_1b)

    def test_different_sc_column(self):
        config_1c = "./tests/scoresfile_tests/config_1c.yaml"
        scores_1c = "./tests/scoresfile_tests/scores_1c.csv"

        pytest.raises(SystemExit, bt4vt.core.SpeakerBiasTest, scores_1c, config_1c)

    def test_different_lab_column(self):
        config_1d = "./tests/scoresfile_tests/config_1d.yaml"
        scores_1d = "./tests/scoresfile_tests/scores_1d.csv"

        pytest.raises(SystemExit, bt4vt.core.SpeakerBiasTest, scores_1d, config_1d)

    def test_additional_column(self):
        # Test Case 2: additional columns we are not using
        config_2 = "./tests/scoresfile_tests/config_2.yaml"
        scores_2 = "./tests/scoresfile_tests/scores_2.csv"

        test_2 = bt4vt.core.SpeakerBiasTest(scores_2, config_2)
        test_2.run_tests()

        assert filecmp.cmp("./tests/scoresfile_tests/results/biastest_results_config_2_scores_2.csv",
                           "./tests/scoresfile_tests/reference_results/reference_biastest_results_config_2_scores_2.csv",
                           shallow=False) == True

    def test_datatype_int(self):
        # Test Case 3a: different data types - ref and com as int
        config_3a = "./tests/scoresfile_tests/config_3a.yaml"
        scores_3a = "./tests/scoresfile_tests/scores_3a.csv"

        test_3a = bt4vt.core.SpeakerBiasTest(scores_3a, config_3a)
        test_3a.run_tests()

        assert filecmp.cmp("./tests/scoresfile_tests/results/biastest_results_config_3a_scores_3a.csv",
                           "./tests/scoresfile_tests/reference_results/reference_biastest_results_config_3a_scores_3a.csv",
                           shallow=False) == True

    def test_datatype_int_str(self):
        # Test Case 3b: different data types - ref as int, com as string
        config_3b = "./tests/scoresfile_tests/config_3b.yaml"
        scores_3b = "./tests/scoresfile_tests/scores_3b.csv"

        test_3b = bt4vt.core.SpeakerBiasTest(scores_3b, config_3b)
        test_3b.run_tests()

        assert filecmp.cmp("./tests/scoresfile_tests/results/biastest_results_config_3b_scores_3b.csv",
                           "./tests/scoresfile_tests/reference_results/reference_biastest_results_config_3b_scores_3b.csv",
                           shallow=False) == True

    def test_datatype_str_int(self):
        # Test Case 3c: different data types - ref as string, com as int
        config_3c = "./tests/scoresfile_tests/config_3c.yaml"
        scores_3c = "./tests/scoresfile_tests/scores_3c.csv"

        test_3c = bt4vt.core.SpeakerBiasTest(scores_3c, config_3c)
        test_3c.run_tests()

        assert filecmp.cmp("./tests/scoresfile_tests/results/biastest_results_config_3c_scores_3c.csv",
                           "./tests/scoresfile_tests/reference_results/reference_biastest_results_config_3c_scores_3c.csv",
                           shallow=False) == True

    def test_id_delimiter(self):
        # Test Case 4: different ways of specifying IDs - "-" instead of "/"
        config_4 = "./tests/scoresfile_tests/config_4.yaml"
        scores_4 = "./tests/scoresfile_tests/scores_4.csv"

        test_4 = bt4vt.core.SpeakerBiasTest(scores_4, config_4)
        test_4.run_tests()

        assert filecmp.cmp("./tests/scoresfile_tests/results/biastest_results_config_4_scores_4.csv",
                           "./tests/scoresfile_tests/reference_results/reference_biastest_results_config_4_scores_4.csv",
                           shallow=False) == True

    def test_labels(self):
        # Test Case 5: different conventions for specifying labels - "-1 and 1" instead of "0 and 1"
        config_5 = "./tests/scoresfile_tests/config_5.yaml"
        scores_5 = "./tests/scoresfile_tests/scores_5.csv"

        test_5 = bt4vt.core.SpeakerBiasTest(scores_5, config_5)
        test_5.run_tests()

        assert filecmp.cmp("./tests/scoresfile_tests/results/biastest_results_config_5_scores_5.csv",
                           "./tests/scoresfile_tests/reference_results/reference_biastest_results_config_5_scores_5.csv",
                           shallow=False) == True
