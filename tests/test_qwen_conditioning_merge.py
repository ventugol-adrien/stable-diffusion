from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from src.nodes.qwen_node import PromptEmbeds, replace_negative_prompt_embeds


class QwenConditioningMergeTests(unittest.TestCase):
    def test_replaces_negatives_with_positive_dtype_and_device(self) -> None:
        positive = PromptEmbeds(
            prompt_embeds=torch.ones((1, 2, 3), dtype=torch.float32),
            pooled_prompt_embeds=torch.ones((1, 4), dtype=torch.float32),
            negative_prompt_embeds=torch.full((1, 2, 3), -1.0, dtype=torch.float32),
            negative_pooled_prompt_embeds=torch.full((1, 4), -1.0, dtype=torch.float32),
        )
        negative = SimpleNamespace(
            negative_prompt_embeds=torch.full((1, 2, 3), 7.0, dtype=torch.float16),
            negative_pooled_prompt_embeds=torch.full((1, 4), 9.0, dtype=torch.float16),
        )

        merged = replace_negative_prompt_embeds(positive, negative)

        self.assertIs(merged.prompt_embeds, positive.prompt_embeds)
        self.assertIs(merged.pooled_prompt_embeds, positive.pooled_prompt_embeds)
        self.assertEqual(merged.negative_prompt_embeds.dtype, torch.float32)
        self.assertEqual(merged.negative_pooled_prompt_embeds.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                merged.negative_prompt_embeds,
                torch.full((1, 2, 3), 7.0, dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                merged.negative_pooled_prompt_embeds,
                torch.full((1, 4), 9.0, dtype=torch.float32),
            )
        )

    def test_raises_when_negative_batch_size_does_not_match(self) -> None:
        positive = PromptEmbeds(
            prompt_embeds=torch.ones((1, 2, 3), dtype=torch.float32),
            pooled_prompt_embeds=torch.ones((1, 4), dtype=torch.float32),
            negative_prompt_embeds=torch.zeros((1, 2, 3), dtype=torch.float32),
            negative_pooled_prompt_embeds=torch.zeros((1, 4), dtype=torch.float32),
        )
        negative = SimpleNamespace(
            negative_prompt_embeds=torch.zeros((2, 2, 3), dtype=torch.float32),
            negative_pooled_prompt_embeds=torch.zeros((2, 4), dtype=torch.float32),
        )

        with self.assertRaisesRegex(RuntimeError, "same batch size"):
            replace_negative_prompt_embeds(positive, negative)


if __name__ == "__main__":
    unittest.main()
