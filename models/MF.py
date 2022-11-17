import torch
from torch import nn
from torch.nn import functional as F


class MF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_user = args.n_user
        self.n_item = args.n_item
        self.d_latent = args.d_latent
        self.sigmoid = nn.Sigmoid()

        self.embeds_u = torch.nn.Embedding(self.n_user, self.latent_dim)
        self.embeds_i = torch.nn.Embedding(self.n_item, self.latent_dim)
        if args.pretrain:
            self.embeds_u.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embeds_i.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
        else:
            # random normal init seems to be better, since LightGCN doesn't use any non-linear activation functions
            if args.init_normal:
                nn.init.normal_(self.embeds_u.weight, std=0.1)
                nn.init.normal_(self.embeds_i.weight, std=0.1)
            else:
                nn.init.xavier_uniform_(self.embeds_u.weight, gain=1)
                nn.init.xavier_uniform_(self.embeds_i.weight, gain=1)

    def get_user_rating(self, users):
        users = users.long()
        users_emb = self.embeds_u(users)
        items_emb = self.embeds_i.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.sigmoid(scores)

    def cal_loss(self, idx_u, idx_i_pos, idx_i_neg):
        emb_u = self.embedding_user(idx_u.long())
        emb_i_pos = self.embedding_item(idx_i_pos.long())
        emb_i_neg = self.embedding_item(idx_i_neg.long())

        pos_scores = torch.sum(emb_u * emb_i_pos, dim=1)
        neg_scores = torch.sum(emb_u * emb_i_neg, dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return loss

    def forward(self, idx_u, idx_i):
        idx_u, idx_i = idx_u.long(), idx_i.long()
        emb_u = self.embeds_u(idx_u)
        emb_i = self.embeds_i(idx_i)
        return self.sigmoid(torch.sum(emb_u * emb_i, dim=1))
