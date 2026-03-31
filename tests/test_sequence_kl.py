# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Tests for sequence-level KL divergence (GitHub issue #21).

Covers get_sequence_logprobs offset slicing, padding mask handling,
KL=0 for identical distributions, and pad_token_id fallback logic.
"""

import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn.functional as F

from heretic.model import Model
from heretic.utils import Prompt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeOutputs:
    """Mimics a HuggingFace model forward-pass output."""

    logits: torch.Tensor


def _make_mock_model(
    vocab_size: int = 10,
    prompt_len: int = 4,
    seq_len: int = 3,
    logits: torch.Tensor | None = None,
) -> MagicMock:
    """Return a mock Model whose internals behave enough for unit tests.

    The mock's ``model`` (the inner HF model) returns *logits* when called.
    The tokenizer produces fixed-length prompt encodings of *prompt_len*.
    """
    mock = MagicMock()

    # --- tokenizer ---
    mock.tokenizer.apply_chat_template.return_value = ["<prompt>"] * 2
    enc = SimpleNamespace(
        input_ids=torch.zeros(2, prompt_len, dtype=torch.long),
        attention_mask=torch.ones(2, prompt_len, dtype=torch.long),
    )
    mock.tokenizer.return_value = enc
    mock.tokenizer.pad_token_id = 0
    mock.tokenizer.eos_token_id = 1

    # --- inner HF model ---
    total_len = prompt_len + seq_len
    if logits is None:
        logits = torch.randn(2, total_len, vocab_size)
    mock.model.return_value = _FakeOutputs(logits=logits)
    mock.model.device = torch.device("cpu")

    # --- settings ---
    mock.settings.batch_size = 32

    # --- attributes used in get_sequence_logprobs ---
    mock.thinking_profile = None
    mock.response_prefix = None

    return mock


# ---------------------------------------------------------------------------
# Test 1: get_sequence_logprobs offset slicing
# ---------------------------------------------------------------------------


class TestSequenceLogprobsOffset:
    def test_get_sequence_logprobs_known_logits_extracts_correct_slice(self) -> None:
        """Verify the offset slicing: logits[prompt_len-1 : prompt_len+seq_len-1]."""
        torch.manual_seed(42)

        vocab_size = 10
        prompt_len = 4
        seq_len = 3
        batch = 2
        total_len = prompt_len + seq_len

        # Build logits with a known pattern so we can verify the slice.
        logits = torch.arange(total_len * vocab_size, dtype=torch.float).unsqueeze(0)
        logits = logits.reshape(1, total_len, vocab_size).expand(batch, -1, -1).clone()

        mock = _make_mock_model(
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            seq_len=seq_len,
            logits=logits,
        )

        prompts = [Prompt(system="sys", user="usr")] * batch
        ref_ids = torch.zeros(batch, seq_len, dtype=torch.long)

        # Call the real method on the mock instance.
        result = Model.get_sequence_logprobs(mock, prompts, ref_ids)

        # Expected slice: positions [prompt_len-1, prompt_len, prompt_len+seq_len-2]
        expected_raw = logits[:, prompt_len - 1 : prompt_len + seq_len - 1, :]
        expected = F.log_softmax(expected_raw, dim=-1)

        assert result.shape == (batch, seq_len, vocab_size)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Test 2: padding mask excludes pad tokens
# ---------------------------------------------------------------------------


class TestPaddingMaskExcludesPadTokens:
    def test_compute_sequence_kl_streaming_masked_positions_excluded_from_average(
        self,
    ) -> None:
        """Masked (pad) positions must not affect the KL average."""
        torch.manual_seed(42)

        batch = 2
        seq_len = 4
        vocab_size = 8

        # Create synthetic base logprobs (uniform distribution in log-space).
        base_logprobs = (
            torch.full((batch, seq_len, vocab_size), 1.0 / vocab_size)
            .log()
            .to(torch.float16)
        )

        # Trial logprobs: slightly different distribution to get nonzero KL.
        trial_raw = torch.randn(batch, seq_len, vocab_size)
        trial_logprobs = F.log_softmax(trial_raw, dim=-1)

        # Mask: only positions 0 and 1 are real; 2 and 3 are padding.
        mask = torch.tensor([[True, True, False, False], [True, True, False, False]])

        # Compute expected KL manually over unmasked positions only.
        base_full = base_logprobs.to(dtype=trial_logprobs.dtype)
        kl_per_pos = F.kl_div(
            trial_logprobs, base_full, reduction="none", log_target=True
        ).sum(dim=-1)
        expected_kl = (kl_per_pos * mask.float()).sum().item() / mask.sum().item()

        # Now test via compute_sequence_kl_streaming with a disk-backed file.
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            shape = (batch, seq_len, vocab_size)
            fp = np.memmap(tmp.name, dtype="float16", mode="w+", shape=shape)
            fp[:] = base_logprobs.numpy()
            fp.flush()
            del fp

            mock = _make_mock_model(vocab_size=vocab_size, seq_len=seq_len)
            # Override get_sequence_logprobs to return our synthetic trial data.
            mock.get_sequence_logprobs = MagicMock(return_value=trial_logprobs)

            prompts = [Prompt(system="sys", user="usr")] * batch
            ref_ids = torch.zeros(batch, seq_len, dtype=torch.long)

            result = Model.compute_sequence_kl_streaming(
                mock,
                prompts,
                ref_ids,
                mask,
                tmp.name,
                shape,
            )

        assert abs(result - expected_kl) < 1e-4, (
            f"KL mismatch: {result} vs {expected_kl}"
        )


# ---------------------------------------------------------------------------
# Test 3: KL ≈ 0 when distributions are identical
# ---------------------------------------------------------------------------


class TestKlZeroWhenIdentical:
    def test_compute_sequence_kl_streaming_identical_distributions_returns_zero(
        self,
    ) -> None:
        """When trial == base logprobs, KL divergence must be approximately 0."""
        torch.manual_seed(42)

        batch = 3
        seq_len = 5
        vocab_size = 8

        # Both base and trial are the same log-softmax distribution.
        raw = torch.randn(batch, seq_len, vocab_size)
        logprobs = F.log_softmax(raw, dim=-1)
        base_logprobs_f16 = logprobs.to(torch.float16)

        # All positions are real (no padding).
        mask = torch.ones(batch, seq_len, dtype=torch.bool)

        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            shape = (batch, seq_len, vocab_size)
            fp = np.memmap(tmp.name, dtype="float16", mode="w+", shape=shape)
            fp[:] = base_logprobs_f16.numpy()
            fp.flush()
            del fp

            mock = _make_mock_model(vocab_size=vocab_size, seq_len=seq_len)
            # Trial returns the same distribution (converted back from f16 for parity).
            mock.get_sequence_logprobs = MagicMock(
                return_value=base_logprobs_f16.float()
            )

            prompts = [Prompt(system="sys", user="usr")] * batch
            ref_ids = torch.zeros(batch, seq_len, dtype=torch.long)

            result = Model.compute_sequence_kl_streaming(
                mock,
                prompts,
                ref_ids,
                mask,
                tmp.name,
                shape,
            )

        # f16 round-trip introduces tiny error; should still be near zero.
        assert abs(result) < 1e-3, f"Expected KL ≈ 0, got {result}"


# ---------------------------------------------------------------------------
# Test 4: pad_token_id fallback to eos_token_id
# ---------------------------------------------------------------------------


class TestPadTokenFallback:
    def test_generate_reference_ids_no_pad_token_uses_eos(self) -> None:
        """generate_reference_ids must fall back to eos_token_id when pad is None."""
        torch.manual_seed(42)

        seq_length = 4
        eos_id = 42

        mock = _make_mock_model()
        mock.tokenizer.pad_token_id = None
        mock.tokenizer.eos_token_id = eos_id

        # generate() returns (inputs, outputs) where outputs includes prompt + gen.
        prompt_len = 3
        gen_len = 2  # shorter than seq_length → padding needed
        total_len = prompt_len + gen_len

        input_ids = torch.zeros(1, prompt_len, dtype=torch.long)
        outputs = torch.zeros(1, total_len, dtype=torch.long)
        # Put recognisable token IDs in generated portion.
        outputs[0, prompt_len:] = torch.tensor([10, 20])

        mock.generate = MagicMock(return_value=({"input_ids": input_ids}, outputs))

        prompts = [Prompt(system="sys", user="usr")]

        ref_ids, ref_mask = Model.generate_reference_ids(mock, prompts, seq_length)

        # Padded positions (indices 2 and 3) should use eos_id.
        assert ref_ids.shape == (1, seq_length)
        assert ref_mask.shape == (1, seq_length)

        # Real tokens at positions 0-1, padding at 2-3.
        assert ref_ids[0, 0].item() == 10
        assert ref_ids[0, 1].item() == 20
        assert ref_ids[0, 2].item() == eos_id
        assert ref_ids[0, 3].item() == eos_id

        # Mask: True for real, False for padding.
        assert ref_mask[0, 0].item() is True
        assert ref_mask[0, 1].item() is True
        assert ref_mask[0, 2].item() is False
        assert ref_mask[0, 3].item() is False
