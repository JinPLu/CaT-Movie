"""定稿渲染器第 3,000 步权重的推理配置：论文表 1 中 CaT-Movie 一行读它。

与该行出片时实际读的 renderer.json 逐字段核过：模型、采样、参考加噪都相同，只差名称、路径与只在训练时用的保存间隔。
"""
from configs.infer._renderer_entity_planning import consumer_config

EXPERIMENT_CONFIG = consumer_config(
    "renderer_wan22_final.py",
    "wan22_final_704p__20261112T231417_893962Z",
    3000,
)
