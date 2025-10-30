
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


class NR_GraphAttention(nn.Module):
    def __init__(self,
                 node_dim,
                 depth=1,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False,
                 device=None):
        super(NR_GraphAttention, self).__init__()

        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()
        self.device = device

        # create parameters
        feature = self.node_dim * (self.depth + 1)

        # gate
        self.gate = torch.nn.Linear(feature, feature)
        torch.nn.init.xavier_uniform_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)

        self.proxy = torch.nn.Parameter(data=torch.empty(128, feature, dtype=torch.float32))  # best
        torch.nn.init.xavier_uniform_(self.proxy)

        # attention kernel
        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)

        self.img_attention_weights = nn.Parameter(torch.randn(128, 128))

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]
        triple_size = inputs[5]
        rel_size = inputs[6]
        node_size = inputs[7]
        mask = inputs[8]
        features = self.activation(features)

        if len(inputs) > 9:
            img_feat = inputs[9]
            img_feat = self.activation(img_feat)

        # print(features.shape)
        outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            neighs = features[adj[1, :].long()]
            if mask == None:
                tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                                  size=[triple_size, rel_size], dtype=torch.float32)
            else:
                tmp = torch.zeros(adj.shape[1]).to(self.device)
                cur = (mask * r_val).float()
                # print(cur)
                tmp = tmp.scatter_add_(0, r_index[0].long(), cur)
                tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                                  size=[triple_size, rel_size], dtype=torch.float32)

            if len(inputs) > 9:
                tri_rel = torch.sparse.mm(tri_rel, img_feat)
            else:
                tri_rel = torch.sparse.mm(tri_rel, rel_emb)

            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            if mask != None:
                neighs = neighs - 2 * (torch.sum(neighs * tri_rel, dim=1, keepdim=True) * tri_rel)
            else:
                neighs = neighs - 2 * torch.sum(neighs * tri_rel, dim=1, keepdim=True) * tri_rel

            # Original
            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)  # att: torch.Size([27793, 27793])
            if mask != None:
                att[tmp == 0] = -1e10

            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[node_size, node_size])
            att = torch.sparse.softmax(att, dim=1)

            if mask != None:
                mask_adj = torch.sparse_coo_tensor(indices=adj, values=tmp, size=[node_size, node_size])
                att = att * mask_adj

            new_features = scatter_sum(src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0, :].long())
            pad = torch.zeros([att.shape[0] - new_features.shape[0], neighs.shape[1]]).to(self.device)
            new_features = torch.cat((new_features, pad), 0)
            features = self.activation(new_features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=-1)
        return outputs
