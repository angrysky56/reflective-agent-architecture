"""
Processor: Transformer Decoder for Sequence Generation

Implements the token-level sequence generation component of RAA.
The Processor is biased by goal states from the Pointer component.
"""

from .goal_biased_attention import GoalBiasedAttention
from .transformer_decoder import ProcessorConfig, TransformerDecoder

__all__ = ["TransformerDecoder", "ProcessorConfig", "GoalBiasedAttention", "Processor"]


# Alias for main interface
class Processor(TransformerDecoder):
    """
    Main interface for the Processor component.

    The Processor generates token sequences, biased by the current goal
    state maintained by the Pointer component.
    """
    pass
