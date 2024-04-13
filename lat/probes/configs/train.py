from dataclasses import dataclass, field

@dataclass
class train_config:
    weight_decay: float=0.001
    model: str='llama2_7b'