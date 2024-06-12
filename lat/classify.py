from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from lat.probes.utils.model_utils import load_probe, get_activations, get_model_tokenizer 

class Classifier:
    def __init__(
        self,
    ):
        pass

    def get_model(self):
        pass

    def classify(self):
        pass


class Llama3Classifier(Classifier):
    def __init__(
        self,
    ):
        self.tokenizer = None
        self.model = None
        super().__init__()

    def get_model(self):
        if self.model is not None:
            return self.tokenizer, self.model
        model_id = "meta-llama/Meta-Llama-Guard-2-8B"
        device = "cuda"
        dtype = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        )
        # model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
        return tokenizer, model

    def classify(self, chat):
        self.tokenizer, self.model = self.get_model()
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(
            "cuda"
        )
        output = self.model(input_ids=input_ids).logits
        safe_id, unsafe_id = self.tokenizer("safe", "unsafe")["input_ids"]
        safe_score = output[0, -1, safe_id].item()
        unsafe_score = output[0, -1, unsafe_id].item()
        safe_probability = torch.nn.functional.softmax(
            torch.tensor([safe_score, unsafe_score]), dim=0
        )[0].item()
        return safe_probability


class ProbeClassifier(Classifier):
    def __init__(self, probe_location):
        self.tokenizer = None
        self.model = None
        self.probe = load_probe(probe_location)
        super().__init__()

    def get_model(self):
        if self.model is not None:
            return self.tokenizer, self.model
        tokenizer, model = get_model_tokenizer("llama2_chat_7b")
        return tokenizer, model

    def classify(self, chat, layer=5):
        self.tokenizer, self.model = self.get_model()
        chat = [{"question": chat}]
        activations = get_activations(self.model, self.tokenizer, chat, layer=layer)
        safe_probability = self.probe.predict(activations)
        print(safe_probability, "safe tensor")
        safe_probability = safe_probability[0][1].item()
        # safe_probability = torch.nn.functional.softmax(
        #     safe_probability, dim=-1
        # )
        # print(safe_probability, "safe")
        # safe_probability = safe_probability[0].item()
        return safe_probability


# moderate([
#     {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
#     {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
# ])
# `safe`
