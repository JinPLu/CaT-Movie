"""Paper Table 1 CaT-Movie inference configuration (Wan2.2-TI2V-5B renderer, step 3,000 checkpoint).

Matches the exact model, attention routing, reference memory, and sampling settings used to generate the benchmark videos.
"""
from configs.infer._renderer_entity_planning import consumer_config

EXPERIMENT_CONFIG = consumer_config(
    "renderer_wan22_final.py",
    "wan22_final_704p__20261112T231417_893962Z",
    3000,
)
