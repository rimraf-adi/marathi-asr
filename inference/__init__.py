"""
Inference module for Marathi Streaming ASR and Baseline Comparison Suite.
Includes complete pipelines for:
- Our Model (Multi-Dialect MoE Conformer with Multi-Exit)
- SraVaani 1.0 (ARTPARK-IISc/SraVaani-1.0)
- IndicConformer 600M (ai4bharat/indic-conformer-600m-multilingual)
"""

from .pipeline import UnifiedASRPipeline, evaluate_all

__all__ = ["UnifiedASRPipeline", "evaluate_all"]
