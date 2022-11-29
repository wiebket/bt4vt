import bt4vt
import filecmp
import pytest


class TestConfigInput:
    def test_different_results_dir(self):
        # Test Case 1: different results directory - without "/" at the end
        config_1 = "./tests/configfile_tests/config_1.yaml"
        scores_1 = "./tests/configfile_tests/scores_1.csv"

        test_1 = bt4vt.core.SpeakerBiasTest(scores_1, config_1)
        test_1.run_tests()
        assert filecmp.cmp("./tests/configfile_tests/results/biastest_results_config_1_scores_1.csv",
                           "./tests/configfile_tests/reference_results/reference_biastest_results_config_1_scores_1.csv",
                           shallow=False) == True

    def test_different_id(self):
        # Test Case 2: different metadata names
        config_2a = "./tests/configfile_tests/config_2a.yaml"
        scores_2a = "./tests/configfile_tests/scores_2a.csv"

        pytest.raises(SystemExit, bt4vt.core.SpeakerBiasTest, scores_2a, config_2a)

    def test_different_select_column(self):
        config_2b = "./tests/configfile_tests/config_2b.yaml"
        scores_2b = "./tests/configfile_tests/scores_2b.csv"

        pytest.raises(SystemExit, bt4vt.core.SpeakerBiasTest, scores_2b, config_2b)

    def test_different_speaker_group(self):
        config_2c = "./tests/configfile_tests/config_2c.yaml"
        scores_2c = "./tests/configfile_tests/scores_2c.csv"

        pytest.raises(SystemExit, bt4vt.core.SpeakerBiasTest, scores_2c, config_2c)

    def test_ptarget(self):
        #PTarget out of range 0-1
        config_3 = "./tests/configfile_tests/config_3.yaml"
        scores_3 = "./tests/configfile_tests/scores_3.csv"

        pytest.raises(Exception, bt4vt.core.SpeakerBiasTest, scores_3, config_3)

    def test_dcf_costs(self):
        # Test Case 4: Multiple DCF Costs
        config_4 = "./tests/configfile_tests/config_4.yaml"
        scores_4 = "./tests/configfile_tests/scores_4.csv"

        test_4 = bt4vt.core.SpeakerBiasTest(scores_4, config_4)
        test_4.run_tests()

        assert filecmp.cmp("./tests/configfile_tests/results/biastest_results_config_4_scores_4.csv",
                           "./tests/configfile_tests/reference_results/reference_biastest_results_config_4_scores_4.csv",
                           shallow=False) == True

    def test_format_select_columns(self):
        # Test Case 5a: Select Colums added as string
        config_5a = "./tests/configfile_tests/config_5a.yaml"
        scores_5a = "./tests/configfile_tests/scores_5a.csv"

        pytest.raises(ValueError, bt4vt.core.SpeakerBiasTest, scores_5a, config_5a)

    def test_format_speaker_groups(self):
        # Test Case 5b: Speaker Groups added as string not as list
        config_5b = "./tests/configfile_tests/config_5b.yaml"
        scores_5b = "./tests/configfile_tests/scores_5b.csv"

        pytest.raises(ValueError, bt4vt.core.SpeakerBiasTest, scores_5b, config_5b)

    def test_format_dcf_costs(self):
        # Test Case 5c: DCF cost added as list not as list of lists
        config_5c = "./tests/configfile_tests/config_5c.yaml"
        scores_5c = "./tests/configfile_tests/scores_5c.csv"

        pytest.raises(ValueError, bt4vt.core.SpeakerBiasTest, scores_5c, config_5c)
