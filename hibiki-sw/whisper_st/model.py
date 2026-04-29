"""TranscriptPromptedWhisper: Whisper + CTC head for transcript-prompted ST.

Architecture (after Wakanda AI / Gichamba et al., Afretec 2025):

    audio -> WhisperEncoder -> hidden states (T x D)
                                    |
                       +------------+-----------+
                       |                        |
                  CTC head                 WhisperDecoder
                  (Linear -> V_ctc)        (cross-attn on encoder + decoder
                       |                    input prefixed with transcript)
                       |                        |
                  CTC loss vs              CE loss vs
                  source transcript        target translation

Joint loss = w_ctc * CTCLoss + w_ce * CELoss

At training time the decoder receives the GROUND-TRUTH transcript prefixed in
its input_ids (no exposure-bias mitigation; the original poster doesn't address
it). At inference time CTC beam-decode produces the transcript hypothesis,
which is then prepended to the decoder prompt for the translation pass.

Decoder input format (training and inference):
    <|startoftranscript|><|sw|><|transcribe|>{transcript_tokens}<|en|><|translate|>{translation_tokens}<|endoftext|>

Loss masking on the decoder side: positions corresponding to special tokens and
the transcript prefix are masked (-100) so the decoder is only supervised on
the translation portion. The CTC loss provides the supervision signal for the
encoder->transcript task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperConfig, WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class TranscriptPromptedOutput(Seq2SeqLMOutput):
    """Extends Seq2SeqLMOutput with a separate CTC loss component for logging."""
    ctc_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    ctc_logits: Optional[torch.FloatTensor] = None


class TranscriptPromptedWhisper(WhisperForConditionalGeneration):
    """Whisper with an extra CTC head on the encoder for transcript supervision.

    Use the same vocabulary as Whisper for the CTC head so we can decode CTC
    output directly with the existing tokenizer. The blank token for CTC is
    the pad token id (0 in Whisper's vocab).
    """

    def __init__(
        self,
        config: WhisperConfig,
        ctc_loss_weight: float = 0.3,
        ce_loss_weight: float = 1.0,
        ctc_blank_id: int = 0,
    ):
        super().__init__(config)
        self.ctc_loss_weight = ctc_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.ctc_blank_id = ctc_blank_id

        # CTC head: project encoder hidden states to Whisper's vocabulary
        self.ctc_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Initialize from output_embeddings if available (warm start)
        with torch.no_grad():
            try:
                out_embed = self.get_output_embeddings().weight
                self.ctc_head.weight.copy_(out_embed)
            except Exception:
                nn.init.normal_(self.ctc_head.weight, std=0.02)

    def forward(
        self,
        input_features: torch.FloatTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        transcript_labels: Optional[torch.LongTensor] = None,
        transcript_label_lengths: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, TranscriptPromptedOutput]:
        """Joint forward pass.

        Args:
            input_features: (B, n_mels, T_in) Whisper mel spectrograms.
            decoder_input_ids: (B, L) full prompt including
                <|sot|><|sw|><|transcribe|>{transcript}<|en|><|translate|>{translation}.
            labels: (B, L) labels for CE loss; positions for non-translation
                tokens should be -100 (ignored).
            transcript_labels: (B, T_max) token IDs of the source transcript
                (no special tokens), padded with -100 for CTC.
            transcript_label_lengths: (B,) actual lengths of transcript_labels.

        Returns:
            TranscriptPromptedOutput with .loss, .ctc_loss, .ce_loss, .logits.
        """
        # 1) Encode audio
        encoder_outputs = self.model.encoder(
            input_features=input_features,
            return_dict=True,
        )
        encoder_hidden = encoder_outputs.last_hidden_state  # (B, T_enc, D)

        # 2) CTC head
        ctc_logits = self.ctc_head(encoder_hidden)  # (B, T_enc, V)

        # 3) Decoder forward (standard Whisper CE head)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden,
            return_dict=True,
        )
        lm_logits = self.proj_out(decoder_outputs.last_hidden_state)  # (B, L, V)

        # 4) Losses
        ctc_loss = None
        if transcript_labels is not None and transcript_label_lengths is not None:
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T_enc, B, V)
            input_lengths = torch.full(
                (ctc_logits.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=ctc_logits.device,
            )
            # Replace -100 (HF padding) with blank for CTC
            transcript_targets = transcript_labels.masked_fill(
                transcript_labels == -100, self.ctc_blank_id
            )
            ctc_loss = F.ctc_loss(
                log_probs,
                transcript_targets,
                input_lengths,
                transcript_label_lengths,
                blank=self.ctc_blank_id,
                reduction="mean",
                zero_infinity=True,
            )

        ce_loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        total_loss = None
        if ctc_loss is not None and ce_loss is not None:
            total_loss = self.ctc_loss_weight * ctc_loss + self.ce_loss_weight * ce_loss
        elif ce_loss is not None:
            total_loss = ce_loss
        elif ctc_loss is not None:
            total_loss = ctc_loss

        return TranscriptPromptedOutput(
            loss=total_loss,
            logits=lm_logits,
            ctc_loss=ctc_loss,
            ce_loss=ce_loss,
            ctc_logits=ctc_logits,
            encoder_last_hidden_state=encoder_hidden,
        )

    @torch.no_grad()
    def ctc_greedy_decode(
        self,
        input_features: torch.FloatTensor,
        tokenizer,
    ) -> list:
        """Greedy CTC decode of the source transcript from the encoder.

        Returns a list of decoded strings, one per batch element.
        """
        self.eval()
        encoder_outputs = self.model.encoder(input_features=input_features, return_dict=True)
        ctc_logits = self.ctc_head(encoder_outputs.last_hidden_state)  # (B, T, V)
        argmax = ctc_logits.argmax(dim=-1)  # (B, T)

        decoded = []
        for seq in argmax:
            # Standard CTC greedy: collapse repeats, drop blanks
            ids = []
            prev = -1
            for tok in seq.tolist():
                if tok != prev and tok != self.ctc_blank_id:
                    ids.append(tok)
                prev = tok
            decoded.append(tokenizer.decode(ids, skip_special_tokens=True))
        return decoded
