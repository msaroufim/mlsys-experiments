from nltk.tokenize import WhitespaceTokenizer

import time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def tokens_to_ids(sentence, vocab):
    tk = WhitespaceTokenizer()
    tokens = tk.tokenize(sentence)
    
    # Map tokens to their corresponding indices in vocab
    ids = [vocab.index(token) if token in vocab else -1 for token in tokens]
    
    return ids

def ids_to_tokens(ids, vocab):
    tokens = [vocab[i] if i < len(vocab) and i >= 0 else "<UNK>" for i in ids]
    
    return tokens

vocab = ["the", "dog", "ran", "across", "the", "street"]
sentence = "ran across the street"
ids = tokens_to_ids(sentence, vocab)
print(ids)

tokens = ids_to_tokens(ids, vocab)
print(tokens)


import torch

def gpt(inputs: list[int]) -> list[list[float]]:
    # inputs has shape [n_seq]
    # output has shape [n_seq, n_vocab]
    output = torch.randn(len(inputs), len(vocab)) # A very dumb gpt
    return output


output = gpt(ids) # output[i][j] probability that token at vocab[j] is the next token after token at inputs[i+1]

# Greedily decode the output for a single token
next_id = torch.argmax(output[-1]) # take the max of the last row
next_word = vocab[next_id]
print(next_word)

def generate(inputs, num_tokens_to_generate=10):
    inputs = ids
    for i in range(num_tokens_to_generate):
        output = gpt(inputs) # Notice that we are loading inputs over and over again with one extra token each time
        next_id = int(torch.argmax(output[-1]))
        inputs.append(next_id)
    return inputs

out = generate(ids)
print(out)


# Can we do better?
# Sketch a high level solution
def generate_with_cache(inputs, num_tokens_to_generate=10):
    cached_outputs = None
    for i in range(num_tokens_to_generate):
        if cached_outputs is not None:
            # Only use the last token for GPT to compute the next logits
            output = gpt(inputs[-1:])
            # Append the new logits to the cached outputs
            cached_outputs = torch.cat((cached_outputs, output), dim=0)
        else:
            # If there's no cached output, compute for the entire input sequence
            cached_outputs = gpt(inputs)
        
        next_id = int(torch.argmax(cached_outputs[-1]))
        inputs.append(next_id)
    
    return inputs[0: len(inputs) - num_tokens_to_generate]

print(generate_with_cache(ids))




# Let's assume we have a transformer with a single head
# Above is equivalent to caching Q, K, V
# But actual attention formula looks like
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return torch.softmax(q @ k.T / torch.sqrt(q.shape[-1]) + mask) @ v


# Transformer crash course
"""
1. Q: Question
2. K: Key to Answer
3. V: Answer
So we only need to cache KV because Q is the last token in the input sequence
"""

# So let's see how this actually implemented https://github.com/jaymody/picoGPT/pull/7/files

# Can we warmup the cache? We don't need to wait until we start decoding to do it
# Most LLMs come with the same prompt so we can precompute the cache for the prompt
# This is called a prefill
def warmup_cache(inputs):
    cache = gpt(inputs)

"""
Read more
Above assumes that we have a single attention head. In practice, we have multiple attention heads.
So should we have a single KV cache for all heads, or a separate KV cache for each head?
1. KV cache for each head: Traditional implementation - accurate but slow
2. Single KV cache for all heads: Multi Query Attention - not accurate but fast
3. k KV caches for n heads where k < n: Group Query Attention - tradeoff between 1 and 2
Can we torch.compile a kv cache, yes! This is horace llama
1. Make kv cache shape static
2. Modify cache in place - see https://github.com/jaymody/picoGPT/pull/7/files#diff-3892986ce3cf6bb007a2ab4ee718a6b349bf0ba3d389edccffe180a95ea48cd0R107
3. Prefill (warmup) the kv cache
vLLM: kv cache can be so large that we need to page it
"""
