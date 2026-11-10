"""Optional CPU-only integration with real PEFT weights and the Model caller."""

import importlib.util
import unittest
from types import SimpleNamespace

from tests import test_vllm_backend


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "requires the project's ML environment"
)
class ModelVllmIntegrationTests(unittest.TestCase):
    def test_every_abliteration_mode_exports_current_weights_through_production_caller(
        self,
    ):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        from heretic.config import RowNormalization
        from heretic.model import AbliterationParameters, Model
        from heretic.utils import Prompt

        fixture = test_vllm_backend.VllmBackendTests()
        fixture.setUp()
        try:
            config = LlamaConfig.from_dict(
                {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "vocab_size": 32,
                    "eos_token_id": 2,
                }
            )
            config._name_or_path = "fixture"
            inner = LlamaForCausalLM(config)
            peft_config = LoraConfig(
                r=1,
                lora_alpha=1,
                target_modules=["o_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            model = object.__new__(Model)
            model.model = get_peft_model(inner, peft_config)
            model.peft_config = peft_config
            model.vllm_backend = fixture.backend
            model.settings = SimpleNamespace(
                model="fixture",
                response_prefix="prefix",
                max_response_length=10,
                vllm=SimpleNamespace(batch_size=2),
            )

            class Tokenizer:
                def apply_chat_template(self, chats, **kwargs):
                    self.template_kwargs = kwargs
                    return [
                        f"{chat[0]['content']}:{chat[1]['content']}:" for chat in chats
                    ]

                def __call__(self, prompts):
                    self.prompts = prompts
                    return {"input_ids": [[i + 1, 3] for i in range(len(prompts))]}

                def batch_decode(self, outputs, **kwargs):
                    self.decode_kwargs = kwargs
                    return [str(tokens) for tokens in outputs]

            tokenizer = Tokenizer()
            model.tokenizer = tokenizer
            directions = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
            for mode in RowNormalization:
                model.settings.row_normalization = mode
                for weight in (1.0, 2.0):
                    parameters = {
                        component: AbliterationParameters(weight, 0, weight, 1)
                        for component in model.get_abliterable_components()
                    }
                    model.abliterate(directions, None, parameters)
                    responses = model.get_responses_batched(
                        [
                            Prompt(system="sys", user="one"),
                            Prompt(system="sys", user="two"),
                        ],
                        skip_special_tokens=True,
                        enable_thinking=False,
                    )
                    self.assertEqual(responses, ["[11, 2]", "[12, 2]"])
                    self.assertTrue(
                        all(p.endswith("prefix") for p in tokenizer.prompts)
                    )
                    self.assertFalse(tokenizer.template_kwargs["enable_thinking"])
                    self.assertTrue(tokenizer.decode_kwargs["skip_special_tokens"])
                self.assertNotEqual(
                    fixture.loaded_weights[-2], fixture.loaded_weights[-1]
                )
            self.assertEqual(len(fixture.loaded_weights), 6)
            self.assertEqual(list(fixture.root.iterdir()), [])
            # A checkpoint can itself be quantized even when the requested
            # settings say "none". Reject it before contacting the engine.
            setattr(inner, "is_loaded_in_4bit", True)
            from heretic.vllm_backend import VllmBackendError

            events = len(fixture.events)
            with self.assertRaises(VllmBackendError):
                model.get_responses_batched([Prompt(system="sys", user="one")])
            self.assertEqual(len(fixture.events), events)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
