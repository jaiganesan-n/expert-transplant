"""Expert transplant: Gemma 3 1B with its FFNs replaced by frozen MobileMoE-M MoE blocks.

Only two projections per transplanted layer train: W_in (Gemma's 1152 -> MobileMoE's
1024) and W_out (1024 -> 1152).
"""

__version__ = "0.5.0"
