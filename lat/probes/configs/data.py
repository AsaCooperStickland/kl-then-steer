from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    domains: List[str] = field(default_factory=lambda: ['training'])
    data_dir: str='datasets/probing'
    activations_dir: str='activations/refusal'
    probe_dir: str='trained_probes/refusal'
    results_dir: str='results/refusal'
    predictions_dir: str='predictions/refusal'
    data_type: str='training_persuasion0.50_refusal0.50' #options: rated_subset_headlines, headllines_coeff: -0.5
    toxic_data: Optional[List[str]] = None # field(default_factory=lambda: ['illegal_activity'])
    normal_data: Optional[List[str]] = None # field(default_factory=lambda: ['ultrachat'])