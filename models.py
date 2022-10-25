from config import args, cprint
import torch
from torch import nn
from torch.nn import functional as F


class PureMF(nn.Module):
    def __init__(self):
        super(PureMF, self).__init__()
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
        emb_u = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        emb_u, emb_i_pos, emb_i_neg = idx_u.long(), idx_i_pos.long(), idx_i_neg.long()

        pos_scores = torch.sum(emb_u * emb_i_pos, dim=1)
        neg_scores = torch.sum(emb_u * emb_i_neg, dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))

        return loss, reg_loss

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, idx_u, idx_i):
        idx_u, idx_i = idx_u.long(), idx_i.long()
        emb_u = self.embeds_u(idx_u)
        emb_i = self.embeds_i(idx_i)
        return self.sigmoid(torch.sum(emb_u * emb_i, dim=1))


class LightGCN(nn.Module):
    def __init__(self, graph):
        super(LightGCN, self).__init__()
        self.graph = graph

        self.n_user = args.n_user
        self.n_item = args.n_item
        self.d_latent = args.d_latent
        self.n_layer = args.n_layer
        self.keep_prob = args.keep_prob
        self.a_split = args.a_split

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

        self.sigmoid = nn.Sigmoid()

    def dropout_x(self, x):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / self.keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def dropout(self):
        if self.a_split:
            graph = []
            for g in self.graph:
                graph.append(self.dropout_x(g))
        else:
            graph = self.dropout_x(self.graph)
        return graph
    
    def aggregate(self):
        all_emb = torch.cat([self.embeds_u.weight, self.embeds_i.weight])
        x_all = [all_emb]

        if self.config['dropout']:
            if self.training:
                g_drop = self.dropout()
            else:
                g_drop = self.graph
        else:
            g_drop = self.graph
        
        for layer in range(self.n_layers):
            if self.a_split:
                temp_emb = []
                for f in range(len(g_drop)):
                    temp_emb.append(torch.sparse.mm(g_drop[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_drop, all_emb)
            x_all.append(all_emb)

        x_out = torch.mean(torch.stack(x_all, dim=1), dim=1)
        xu_all, xi_all = torch.split(x_out, [self.n_user, self.n_item])
        return xu_all, xi_all
    
    def get_user_rating(self, idx_u):
        xu_all, xi_all = self.aggregate()
        return self.sigmoid(torch.matmul(xu_all[idx_u.long()], xi_all.T))
    
    def get_embedding(self, idx_u, idx_i_pos, idx_i_neg):
        xu_all, xi_all = self.aggregate()
        xu_pos = xu_all[idx_u]
        xi_pos = xi_all[idx_i_pos]
        xi_neg = xi_all[idx_i_neg]

        emb_u_pos = self.embeds_u(xu_pos)
        emb_i_pos = self.embeds_i(xi_pos)
        emb_i_neg = self.embeds_i(xi_neg)

        return xu_pos, xi_pos, xi_neg, emb_u_pos, emb_i_pos, emb_i_neg
    
    def cal_loss(self, idx_u, idx_i_pos, idx_i_neg):
        xu_pos, xi_pos, xi_neg, emb_u_pos, emb_i_pos, emb_i_neg = \
            self.get_embedding(idx_u.long(), idx_i_pos.long(), idx_i_neg.long())

        pos_scores = torch.mul(xu_pos, xi_pos).sum(dim=1)
        neg_scores = torch.mul(xu_pos, xi_neg).sum(dim=1)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        loss_reg = 0.5 * (emb_u_pos.norm(2).pow(2) + emb_i_pos.norm(2).pow(2)
                          + emb_i_neg.norm(2).pow(2)) / float(len(idx_u))

        return loss, loss_reg
       
    def forward(self, idx_u, idx_i):
        xu_all, xi_all = self.aggregate()
        return torch.sum(torch.mul(xu_all[idx_u], xi_all[idx_i]), dim=1)
