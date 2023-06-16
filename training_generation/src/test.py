import unittest
from prompting.prompt_module import PromptModule
import hydra
from omegaconf import DictConfig, OmegaConf


class LangChainTesting(unittest.TestCase):
    def test_one_prompt_has_to_be_defined(self):
        with self.assertRaises(ValueError):
            PromptModule("sarcasm")

    def test_other_inputs(self):
        with self.assertRaises(NotImplementedError):
            PromptModule("s")

    def test_hydra_config(self):
        val = 0

        @hydra.main(version_base="1.2", config_path="conf/", config_name="testing")
        def test_values(cfg: DictConfig):
            self.assertEqual(cfg.val, 42)

        test_values()


if __name__ == '__main__':
    unittest.main()
