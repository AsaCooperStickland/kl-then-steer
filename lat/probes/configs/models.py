from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()
HF_API_KEY = os.getenv("HF_TOKEN")


@dataclass
class model_config:
    weight_decay: float=0.001
    model: str='llama2_chat_7b'
    device: str='cuda'
    seed: int=42
    single_topic_probe: bool=True
    hold_one_out_probe: bool=True
    mixed_probe: bool = True 
    get_predictions: bool = True
    hf_token: str=HF_API_KEY

model_lookup = {'gpt2_medium':{'tl_name' : "gpt2-medium",
                                 'layers' : [4, 8, 12, 16, 20, 23]
                                  },
                    'llama2_7b':{'hf_name' : 'meta-llama/Llama-2-7b-hf',
                                 'tl_name' : "llama-7b",
                                 'layers' : [5, 10, 15, 20, 25, 31]
                                  },
                    'llama2_chat_7b':{'hf_name' : 'meta-llama/Llama-2-7b-chat-hf',
                                 'tl_name' : "meta-llama/Llama-2-7b-chat-hf",
                                 'layers' : [5, 10, 15, 20, 25, 31]
                                  },
                    'llama2_13b':{'hf_name' : 'meta-llama/Llama-2-13b-hf',
                                  'tl_name' : "llama-13b",
                                  'layers' : [5, 10, 15, 20, 25, 30, 35, 39]
                                  },
                    'qwen_1.8b':{'hf_name' : 'Qwen/Qwen-1_8B',
                                  'tl_name' : "qwen-1.8b",
                                  'layers' : [4, 8, 12, 16, 20, 23]
                                  },
                    'qwen_7b':{'hf_name' : 'Qwen/Qwen-7B',
                                  'tl_name' : "qwen-7b",
                                  'layers' : [5, 10, 15, 20, 25, 31]
                                  },
                    'qwen_14b':{'hf_name' : 'Qwen/Qwen-14B',
                                  'tl_name' : "qwen-14b",
                                  'layers' : [5, 10, 15, 20, 25, 30, 35, 39]
                                  },
                    'pythia_160m':{'hf_name' : 'EleutherAI/pythia-160m',
                                  'tl_name' : "pythia-160m",
                                  'layers' : [2, 4, 6, 8, 10, 11]
                                  },
                    'pythia_1.4b':{'hf_name' : 'EleutherAI/pythia-1.4b',
                                  'tl_name' : "pythia-1.4b",
                                  'layers' : [4, 8, 12, 16, 20, 23]
                                  },
                    'pythia_6.9b':{'hf_name' : 'EleutherAI/pythia-6.9b',
                                  'tl_name' : "pythia-6.9b",
                                  'layers' : [5, 10, 15, 20, 25, 31]
                                  },
                    'mistral_7b': {'hf_name': 'mistralai/Mistral-7B-v0.1',
                                   'tl_name': "mistralai/Mistral-7B-v0.1",
                                   'layers' : [5, 10, 15, 20, 25, 31]}
                    }