import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from transformers import AutoTokenizer, AutoModelForCausalLM


# def add_vector_after_position(matrix, vector, position_ids, after=None):
#     after_id = after
#     if after_id is None:
#         after_id = position_ids.min().item() - 1
#     mask = position_ids > after_id
#     mask = mask.unsqueeze(-1)
#     matrix += mask.float() * vector
#     return matrix

def add_vector_after_position(matrix, vector, position_ids, after=None):
    # Get the matrix shape and broadcast dimensions
    batch_size, seq_len, _ = matrix.size()

    # If after is None, create a default tensor with values smaller than min of position_ids
    if after is None:
        after_val = position_ids.min().item() - 1
        after = torch.full((batch_size, seq_len), after_val).to(position_ids.device)

    # Convert after to tensor if it's a list
    elif isinstance(after, list):
        after = torch.tensor(after).to(position_ids.device).unsqueeze(-1)
        after = after.expand(batch_size, seq_len)

    else:
        after = torch.full((batch_size, seq_len), after).to(position_ids.device)

    # Check if each position in position_ids is greater than the corresponding value in after
    mask = position_ids > after
    mask = mask.unsqueeze(-1).expand_as(matrix)
    matrix += mask.float() * vector
    return matrix


def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    end_pos = find_subtensor_position(tokens, end_str)
    return end_pos + len(end_str) - 1


class LinearWrapper(torch.nn.Module):
    
    def __init__(self, proj):
        super().__init__()
        self.proj = proj
        self.activations = None
        self.add_activations = None

    def forward(self, *args, **kwargs):
        output = self.proj(*args, **kwargs)
        self.activations = output.clone()
        if self.add_activations is not None:
            # output[:, -1, :] = output[:, -1, :] + self.add_activations.reshape(1, 1, -1)
            output += self.add_activations
        return output
    
    def reset(self):
        self.activations = None
        self.add_activations = None
        
    def add(self, activations):
        self.add_activations = activations


class AttnWrapper(torch.nn.Module):
    
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.attn.q_proj = LinearWrapper(self.attn.q_proj)
        self.attn.v_proj = LinearWrapper(self.attn.v_proj)

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            output = (augmented_output + self.add_activations,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.block.self_attn.activations = None
        self.block.self_attn.attn.q_proj.reset()
        self.block.self_attn.attn.v_proj.reset()

        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []

    def get_query_activations(self):
        return self.block.self_attn.attn.q_proj.activations
    
    def get_value_activations(self):
        return self.block.self_attn.attn.v_proj.activations
    
    def add_query_activations(self, activations):
        self.block.self_attn.attn.q_proj.add(activations)
    
    def add_value_activations(self, activations):
        self.block.self_attn.attn.v_proj.add(activations)
        

class Llama7BChatHelper:
    def __init__(self, token, system_prompt, generation=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        if generation:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token
        ).to(self.device)
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(
            self.device
        )
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.model.model.layers:
            layer.after_position = pos

    def prompt_format(self, instruction):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + self.system_prompt + E_SYS + instruction.strip()
        dialog_content = f"{B_INST} {dialog_content.strip()} {E_INST}"
        return dialog_content

    def tokenize_prompt(self, prompt):
        dialog_tokens = self.tokenizer(prompt, padding=True, return_tensors="pt"
        )
        return dialog_tokens

    def generate_text(self, prompts, max_new_tokens=50):
        if type(prompts) == str:
            prompts = [prompts]
        prompts = [self.prompt_format(prompt) for prompt in prompts]
        tokens = self.tokenize_prompt(prompts).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate(self, tokens, max_new_tokens=50):
        instr_pos = [find_instruction_end_postion(tokens.input_ids[i], self.END_STR) for i in range(len(tokens.input_ids))]
        self.set_after_positions(instr_pos)
        generated = self.model.generate(
            **tokens, max_new_tokens=max_new_tokens, top_k=1
        )
        if len(generated) > 1:
            return self.tokenizer.batch_decode(generated)
        else:
            return self.tokenizer.batch_decode(generated)[0]

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def get_last_query_activations(self, layer):
        return self.model.model.layers[layer].get_query_activations()

    def set_add_query_activations(self, layer, activations):
        self.model.model.layers[layer].add_query_activations(activations)
        
    def get_last_value_activations(self, layer):
        return self.model.model.layers[layer].get_value_activations()

    def set_add_value_activations(self, layer, activations):
        self.model.model.layers[layer].add_value_activations(activations)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))
