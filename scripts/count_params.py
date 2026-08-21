def count_params():

    print("Type parameteres in this order (All values should be seperated by space): \n -> vocab_size \n -> context_length \n -> n_embed \n -> num_heads \n -> num_layers")
    vocab_size, context_lenght, n_embed, num_heads, num_layers = [int(x) for x in input().split()]

    head_size = n_embed // num_heads

    x = 3 * head_size * (n_embed + 1)                               # Single Head Attn 
    x = (num_heads * x) + (n_embed ** 2) + n_embed                  # Multi Head Attn
    x = x + (8 * n_embed ** 2) + (5 * n_embed)                      # FeedForward
    x = x + (4 * n_embed)                                           # layer norm 
    x = num_layers * x                                              # Blocks
    x = x + (vocab_size * n_embed) + (context_lenght * n_embed)     # Emdeddings (Token + position)
    x = x + (2 * n_embed)                                           # Final LayerNorm
    x = x + (vocab_size * n_embed) + vocab_size                     # Final Step

    return x

if __name__ == "__main__":
    print(count_params() / 10**6, 'million')