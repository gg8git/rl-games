"""
belief_net.py
────────
GongZhuBeliefPredictor — updated from the original to add:
  • seq_lengths parameter in forward()
  • src_key_padding_mask passed to the TransformerEncoder
  • key_padding_mask passed to the cross-attention module

These changes prevent padding tokens (in batched, variable-length sequences)
from polluting the attention computation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- onnx export ---
def export_to_onnx(pt_path: str, onnx_path: str):
    print(f"Loading PyTorch model from {pt_path}...")
    device = torch.device("cpu")
    
    # Initialize model with the exact same architecture
    model = GongZhuBeliefPredictor(d_model=128, n_heads=4, n_layers=3)
    
    # Handle both raw state_dict and checkpoint dict formats
    checkpoint = torch.load(pt_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # --- Static Shape of 52 ---
    B, T = 1, 52
    dummy_played_cards = torch.zeros((B, T), dtype=torch.long)
    dummy_players      = torch.zeros((B, T), dtype=torch.long)
    dummy_trick_nums   = torch.zeros((B, T), dtype=torch.long)
    dummy_trick_pos    = torch.zeros((B, T), dtype=torch.long)
    dummy_revealed     = torch.zeros((B, 4, 52), dtype=torch.float32)
    dummy_mask         = torch.zeros((B, 52, 4), dtype=torch.float32)
    dummy_seq_lengths  = torch.tensor([10], dtype=torch.long) # Tells ONNX to expect the seq_length tensor

    print("Tracing and exporting to ONNX...")
    torch.onnx.export(
        model,
        args=(
            dummy_played_cards, dummy_players, dummy_trick_nums, dummy_trick_pos,
            dummy_revealed, dummy_mask, dummy_seq_lengths
        ),
        f=onnx_path,
        export_params=True,
        opset_version=14,  # Stable opset for Transformers
        do_constant_folding=True,
        input_names=[
            "played_cards", "players", "trick_nums", "trick_pos", 
            "revealed_hands", "mask", "seq_lengths"
        ],
        output_names=["probabilities", "safe_logits"],
        # dynamic_axes has been completely removed to enforce static graphs
    )
    print(f"Success! Model saved to {onnx_path}")


# --- model ---
class GongZhuBeliefPredictor(nn.Module):
    """
    Predicts, for every card in the deck, the probability distribution over
    which of the 4 players currently holds it.

    Input tensors (all batched with leading dimension B)
    ─────────────────────────────────────────────────────
    played_cards   (B, T)        card indices 0–51 in chronological play order
    players        (B, T)        player index 0–3 who played each card
    trick_nums     (B, T)        trick index 0–12 for each play
    trick_pos      (B, T)        position within the trick 0–3 for each play
    revealed_hands (B, 4, 52)    observer's positive knowledge:
                                   1 = definitely held by that player
                                   0 = unknown
    mask           (B, 52, 4)    additive logit mask:
                                   -inf = card cannot be held by that player
                                    0.0 = no constraint
    seq_lengths    (B,) int64    actual history lengths before padding.
                                 Pass None when all sequences have the same
                                 length (no padding required).

    Output
    ──────
    probabilities   (B, 52, 4)   softmax probabilities; fully-masked cards are 0.
    masked_logits   (B, 52, 4)   logits after applying the mask (used for loss).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 3):
        super().__init__()
        self.d_model = d_model

        # ── 1. History Sequence Embeddings ────────────────────────────────────
        self.card_emb      = nn.Embedding(52, d_model)
        self.player_emb    = nn.Embedding(4,  d_model)
        self.trick_pos_emb = nn.Embedding(4,  d_model)

        # Trick number embedding with sinusoidal inductive bias (fine-tunable)
        self.trick_num_emb = nn.Embedding(13, d_model)
        self._init_sinusoidal(self.trick_num_emb.weight.data, 13, d_model)

        self.history_proj = nn.Linear(d_model * 4, d_model)

        # ── 2. Shared Rule Embeddings ─────────────────────────────────────────
        # revealed_hands values {0, 1} are shifted to {1, 2} before indexing
        # (index 0 is reserved but unused)
        self.revealed_emb = nn.Embedding(3, d_model // 4)

        # ── 3. Global Context (Dynamic START Token) ───────────────────────────
        # Consumes the entire flattened embedded rule state: 52 cards × d_model
        self.start_proj = nn.Sequential(
            nn.Linear(52 * d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # ── 4. Memory Bank (Self-Attention over history) ─────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── 5. Card Queries (Cross-Attention) ─────────────────────────────────
        # Card-aware MLP: card identity (d_model) + embedded rules (d_model)
        self.query_proj_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )

        # ── 6. Output Projection ──────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, 4)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _init_sinusoidal(weight: torch.Tensor, max_len: int, d_model: int) -> None:
        """Inject a sinusoidal prior so adjacent trick numbers start close."""
        pos      = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        weight[:, 0::2] = torch.sin(pos * div_term)
        weight[:, 1::2] = torch.cos(pos * div_term)

    @staticmethod
    def _build_padding_mask(
        seq_lengths: torch.Tensor,   # (B,)  actual history lengths
        total_len:   int,            # 1 + padded_T  (includes START token)
        device:      torch.device,
    ) -> torch.Tensor:
        """
        Build src_key_padding_mask of shape (B, total_len).

        True  → this position is padding and should be ignored by attention.
        False → real token.

        Layout: position 0 is the START token (never padding),
                positions 1..T correspond to history plays.
        """
        # Position i (0-indexed) is padding when i > seq_len (START occupies 0).
        positions = torch.arange(total_len, device=device).unsqueeze(0)   # (1, L)
        pad_mask  = positions > seq_lengths.unsqueeze(1)                  # (B, L)
        pad_mask[:, 0] = False   # START token is always valid
        return pad_mask

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        played_cards:   torch.Tensor,              # (B, T)
        players:        torch.Tensor,              # (B, T)
        trick_nums:     torch.Tensor,              # (B, T)
        trick_pos:      torch.Tensor,              # (B, T)
        revealed_hands: torch.Tensor,              # (B, 4, 52)
        mask:           torch.Tensor,              # (B, 52, 4)
        seq_lengths:    torch.Tensor | None = None # (B,)  optional
    ):
        B, T  = played_cards.shape
        device = played_cards.device

        # ── Pre-compute: embed rule state ─────────────────────────────────────
        # Shift {0, 1} → {1, 2} so we never use embedding index 0.
        revealed_indices  = (revealed_hands.transpose(1, 2) + 1).long()  # (B, 52, 4)
        revealed_embedded = self.revealed_emb(revealed_indices)           # (B, 52, 4, d//4)

        # ── Phase 1: Build chronological memory ───────────────────────────────
        if T > 0:
            h = torch.cat([
                self.card_emb(played_cards),       # (B, T, d)
                self.player_emb(players),           # (B, T, d)
                self.trick_num_emb(trick_nums),     # (B, T, d)
                self.trick_pos_emb(trick_pos),      # (B, T, d)
            ], dim=-1)                              # (B, T, d*4)
            history_seq = self.history_proj(h)      # (B, T, d)
        else:
            history_seq = torch.empty(B, 0, self.d_model, device=device)

        # Dynamic START token from the globally flattened rule embeddings
        flat_revealed  = revealed_embedded.view(B, -1)                    # (B, 52*d)
        start_content  = self.start_proj(flat_revealed).unsqueeze(1)      # (B, 1, d)

        full_sequence  = torch.cat([start_content, history_seq], dim=1)   # (B, 1+T, d)

        # Build padding mask (only when sequences in the batch differ in length)
        if seq_lengths is not None:
            pad_mask = self._build_padding_mask(seq_lengths, 1 + T, device)
        else:
            pad_mask = None

        memory = self.history_encoder(
            full_sequence,
            src_key_padding_mask=pad_mask,          # (B, 1+T) or None
        )                                            # (B, 1+T, d)

        # ── Phase 2: Build card queries and cross-attend to memory ────────────
        all_cards     = torch.arange(52, device=device).unsqueeze(0).expand(B, -1)
        base_queries  = self.card_emb(all_cards)                          # (B, 52, d)

        # Flatten the 4 players' embedded status per card into d_model dims
        revealed_flat = revealed_embedded.view(B, 52, self.d_model)       # (B, 52, d)

        combined_queries = torch.cat([base_queries, revealed_flat], dim=-1) # (B, 52, 2d)
        queries          = self.query_proj_mlp(combined_queries)           # (B, 52, d)

        card_representations, _ = self.cross_attention(
            query=queries,
            key=memory,
            value=memory,
            key_padding_mask=pad_mask,              # mask padding in keys too
            need_weights=False,
        )                                            # (B, 52, d)

        # ── Phase 3: Output and autograd-safe masking ─────────────────────────
        logits        = self.output_proj(card_representations)             # (B, 52, 4)
        masked_logits = logits + mask

        # Safe softmax: cards with every player masked are set to uniform zero
        all_masked = (mask < -1e8).all(dim=-1, keepdim=True).expand_as(masked_logits)
        safe_logits = torch.where(all_masked, torch.zeros_like(masked_logits), masked_logits)
        probabilities = F.softmax(safe_logits, dim=-1)

        # Re-zero probabilities for fully-masked cards (no in-place ops)
        probabilities = torch.where(all_masked, torch.zeros_like(probabilities), probabilities)

        return probabilities, safe_logits