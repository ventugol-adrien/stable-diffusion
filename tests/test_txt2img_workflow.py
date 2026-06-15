from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

import main


class Txt2ImgWorkflowTests(unittest.TestCase):
    def test_txt2img_workflow_uses_compel_negatives(self) -> None:
        request = types.SimpleNamespace(
            user_input="sunset beach",
            negative_input="bad anatomy",
            model="juggernaut",
        )
        qwen_embeds = types.SimpleNamespace(
            prompt_embeds=torch.ones((1, 2, 3), dtype=torch.float32),
            pooled_prompt_embeds=torch.ones((1, 4), dtype=torch.float32),
            negative_prompt_embeds=torch.full((1, 2, 3), 9.0, dtype=torch.float32),
            negative_pooled_prompt_embeds=torch.full((1, 4), 9.0, dtype=torch.float32),
        )
        compel_embeds = types.SimpleNamespace(
            prompt_embeds=torch.full((1, 2, 3), 7.0, dtype=torch.float32),
            pooled_prompt_embeds=torch.full((1, 4), 7.0, dtype=torch.float32),
            negative_prompt_embeds=torch.full((1, 2, 3), 5.0, dtype=torch.float32),
            negative_pooled_prompt_embeds=torch.full((1, 4), 5.0, dtype=torch.float32),
        )
        qwen_node = mock.Mock(return_value=qwen_embeds)
        compel_node = mock.Mock(return_value=compel_embeds)

        with mock.patch("main.QwenNode", return_value=qwen_node) as qwen_ctor:
            with mock.patch("main.CompelNode", return_value=compel_node):
                embeds = main._build_txt2img_conditioning(request)

        qwen_inputs = qwen_ctor.call_args.args[0]

        self.assertFalse(qwen_inputs.qwen_use_cached_negative_prompt_embeds)
        self.assertTrue(torch.equal(embeds.prompt_embeds, qwen_embeds.prompt_embeds))
        self.assertTrue(
            torch.equal(embeds.pooled_prompt_embeds, qwen_embeds.pooled_prompt_embeds)
        )
        self.assertTrue(
            torch.equal(
                embeds.negative_prompt_embeds,
                compel_embeds.negative_prompt_embeds,
            )
        )
        self.assertTrue(
            torch.equal(
                embeds.negative_pooled_prompt_embeds,
                compel_embeds.negative_pooled_prompt_embeds,
            )
        )


if __name__ == "__main__":
    unittest.main()
