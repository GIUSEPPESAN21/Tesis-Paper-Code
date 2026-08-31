"""Neural network components: the GIN graph encoder and the PPO actor-critic policy."""

from __future__ import annotations

from models.gin import (
    GINEncoder,
    HeteroGINEncoder,
    HeteroRelationalGINLayer,
    RelationalGINLayer,
    build_mlp,
)
from models.policy import (
    JSSPGraphFeaturesExtractor,
    JSSPHeteroGraphFeaturesExtractor,
    PerNodeActorHead,
    PerNodeMaskableActorCriticPolicy,
    PerPairActorHead,
    PooledCriticHead,
    build_hetero_maskable_ppo,
    build_maskable_ppo,
)

__all__ = [
    "GINEncoder",
    "HeteroGINEncoder",
    "HeteroRelationalGINLayer",
    "JSSPGraphFeaturesExtractor",
    "JSSPHeteroGraphFeaturesExtractor",
    "PerNodeActorHead",
    "PerNodeMaskableActorCriticPolicy",
    "PerPairActorHead",
    "PooledCriticHead",
    "RelationalGINLayer",
    "build_hetero_maskable_ppo",
    "build_maskable_ppo",
    "build_mlp",
]
