import torch


#         prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
 #        past_key_values = self.prefix_encoder(prefix_tokens)
 #        add mask

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    # def __init__(self, config):
    #     super().__init__()
    #     self.prefix_projection = config.prefix_projection
    #     if self.prefix_projection:
    #         # Use a two-layer MLP to encode the prefix
    #         self.embedding = torch.nn.Embedding(config.prefix_length, config.encoder_embed_dim)
    #         self.trans = torch.nn.Sequential(
    #             torch.nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim),
    #             torch.nn.Tanh(),
    #             torch.nn.Linear(config.encoder_embed_dim, config.encoder_layers * 2 * config.encoder_embed_dim)
    #         )
    #     else:
    #         self.embedding = torch.nn.Embedding(config.prefix_length, config.encoder_layers * 2 * config.encoder_embed_dim)

    # def forward(self, prefix: torch.Tensor):
    #     if self.prefix_projection:
    #         prefix_tokens = self.embedding(prefix)
    #         past_key_values = self.trans(prefix_tokens)
    #     else:
    #         past_key_values = self.embedding(prefix)
    #     return past_key_values
    

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = False
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.prefix_length, config.encoder_embed_dim)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(config.encoder_embed_dim, config.encoder_layers * 2 * config.encoder_embed_dim)
            )
        else:
            # prefix_tokens_num = 
            self.embedding = torch.nn.Embedding(config.prefix_length, config.encoder_layers * 2 * config.encoder_embed_dim)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    



# test = PrefixEncoder("123")
# print(test.state_dict()["embedding.weight"])
# # for name in test.state_dict():
#    print(name)

# for name in test.state_dict():
#    print(name)
#    print(test.state_dict()[name])
# print(test.state_dict()["embedding.weights"])
# print(12345)

