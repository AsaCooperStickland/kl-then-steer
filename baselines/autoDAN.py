import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM)
from tqdm import tqdm

SOFTMAX_FINAL = nn.Softmax(dim=-1)
LOGSOFTMAX_FINAL = nn.LogSoftmax(dim=-1)
CROSSENT = nn.CrossEntropyLoss(reduction='none')

def token_gradients_with_output(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    embed_weights = list(model.modules())[2]
    assert type(embed_weights).__name__=='Embedding'
    embed_weights = embed_weights.weight
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:],
            input_embeds,
            embeds[:,input_slice.stop:,:]
        ],
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    loss.backward()

    return one_hot.grad.clone(), logits.detach().clone()


class autoDAN:
    """
    AutoDAN c.f. https://openreview.net/pdf?id=rOiymxm8tQ
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch: int = 256,
        max_steps: int = 500,
        weight_1: float = 3,
        weight_2: float = 100,
        temperature: float = 1,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.batch = batch
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.temperature = temperature
        self.max_steps = max_steps


    def sample_model(
        self,
        input_ids,
    ):
        logits = self.model(input_ids).logits[:, -1, :]
        probs = SOFTMAX_FINAL(logits/self.temperature)
        samples = torch.multinomial(probs, 1)
        return samples
    

    def optimize(
        self,
        user_query,
        num_tokens,
        target_string,
        connection_tokens=torch.tensor([]),
        stop_its=1,
        verbose=False,
    ):
        '''
        user_query: string, query to be used as input to model. Should include BOS token (in string form) and chat dialogue preamble
        connection_tokens: tensor of tokens to be used as connection tokens for e.g. dialogue. Usually length 0 tensor for base LMs (must not have BOS token)
        
        Use stop_its to avoid early stopping with small batch sizes
        Note BOS token and custom dialogue templates not implemented yet
        Possible problem: tokenization iteratively adds tokens as characters, but these may be tokenized differently after addition
        '''
        query = self.tokenizer.encode(user_query, return_tensors="pt", add_special_tokens=False)[0].cuda()
        query_len = query.shape[-1]
        targets = self.tokenizer.encode(target_string, return_tensors="pt", add_special_tokens=False)[0].cuda()
        batch_targets = targets.unsqueeze(0).repeat(self.batch,1).contiguous()

        initial_x = self.sample_model(query.unsqueeze(0))[0]
        input_ids = torch.cat([query, initial_x, connection_tokens, targets], dim=0)
        adversarial_sequence = []
        adversarial_seq_tensor = torch.tensor(adversarial_sequence,dtype=torch.long).cuda()

        for ind in range(num_tokens): #iteratively construct adversarially generated sequence
            curr_token = query_len + ind
            optimized_slice = slice(curr_token, curr_token+1)
            target_slice = slice(curr_token + 1 + len(connection_tokens), input_ids.shape[-1])
            loss_slice = slice(curr_token + len(connection_tokens), input_ids.shape[-1] - 1)
            # print(f"slices: optimized slice {optimized_slice}, target slice {target_slice}, loss slice {loss_slice}")
            best_tokens = set()
            if verbose:
                print(f"For seq #{self.tokenizer.decode(input_ids)}#")
                print(f"Optimizing token {ind} at index {curr_token}: {self.tokenizer.decode(input_ids[curr_token])}")
            stop = 0
            for step in tqdm(range(self.max_steps)): #optimize current token
                grads, logits = token_gradients_with_output(self.model, input_ids, optimized_slice, target_slice, loss_slice)
                curr_token_logprobs = LOGSOFTMAX_FINAL(logits[0, curr_token-1, :])
                candidate_tokens = torch.topk(-1*self.weight_1*grads+curr_token_logprobs, self.batch-1, dim=-1).indices.detach()
                candidate_tokens = torch.cat((candidate_tokens[0],input_ids[curr_token:curr_token+1]),dim=0) #append previously chosen token
                candidate_sequences = input_ids.unsqueeze(0).repeat(self.batch,1).contiguous()
                candidate_sequences[:,curr_token] = candidate_tokens
                with torch.no_grad():
                    all_logits = self.model(candidate_sequences).logits
                    loss_logits = all_logits[:, loss_slice, :].contiguous()
                    target_losses = CROSSENT(loss_logits.view(-1,loss_logits.size(-1)), batch_targets.view(-1)) #un-reduced cross-ent
                    target_losses = torch.mean(target_losses.view(self.batch,-1), dim=1) #keep only batch dimension
                    combo_scores = -1*self.weight_2*target_losses + curr_token_logprobs[candidate_tokens]
                    combo_probs = SOFTMAX_FINAL(combo_scores/self.temperature)
                    temp_token = candidate_tokens[torch.multinomial(combo_probs, 1)]
                    best_token = candidate_tokens[torch.argmax(combo_probs).item()]
                if step == 0 and verbose:
                    print(f"max prob {torch.max(combo_probs):.2f} temp_token {self.tokenizer.decode(temp_token)} token_id {temp_token.item()} and best_token {self.tokenizer.decode(best_token)} token_id {best_token}")
                    print(f"10 candidate tokens at step {step}: {self.tokenizer.decode(candidate_tokens[:10])}")
                    print("Losses:")
                    print(f"     Initial loss at step {ind}, iteration {step}: {torch.max(-1*target_losses).item():.2f}")
                    print(f"     Combination scores at step {ind}, iteration {step}: {[round(val.item(),2) for val in torch.topk(combo_scores,5)[0]]}") #torch.topk(combo_scores,5)
                elif best_token in best_tokens:
                    stop+=1
                    if stop==stop_its:
                        if verbose:
                            print("Losses:")
                            print(f"     Final loss at step {ind}, iteration {step}: {torch.max(-1*target_losses).item():.2f}")
                            print(f"     Combination scores at step {ind}, iteration {step}: {[round(val.item(),2) for val in torch.topk(combo_scores,5)[0]]}") #torch.topk(combo_scores,5)
                            print(f"Best token {self.tokenizer.decode(best_token)} Sampled token {self.tokenizer.decode(temp_token)}")
                        adversarial_sequence.append(temp_token)
                        break
                else: 
                    best_tokens.add(best_token)
                if step==self.max_steps-1:
                    if verbose:
                        print("Losses:")
                        print(f"     Final loss at step {ind}, iteration {step}: {torch.max(-1*target_losses).item():.2f}")
                        print(f"     Combination scores at step {ind}, iteration {step}: {[round(val.item(),2) for val in torch.topk(combo_scores,5)[0]]}") #torch.topk(combo_scores,5)
                        print(f"Best token {self.tokenizer.decode(best_token)}. Sampled token {self.tokenizer.decode(temp_token)}.")
                    adversarial_sequence.append(temp_token)
                    break
                input_ids = torch.cat([query, adversarial_seq_tensor, temp_token, connection_tokens, targets], dim=0)
            
            adversarial_seq_tensor = torch.tensor(adversarial_sequence,dtype=torch.long).cuda()
            next_in = torch.cat([query, adversarial_seq_tensor],dim=0).unsqueeze(0).type(torch.long)
            next_tok_rand = self.sample_model(next_in)[0]
            input_ids = torch.cat([query, adversarial_seq_tensor, next_tok_rand, connection_tokens, targets], dim=0)


        # print('Final target logprob was:', torch.max(-1*target_losses).item())
        return self.tokenizer.decode(torch.tensor(adversarial_sequence))