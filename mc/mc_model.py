import sys
import os
sys.path.append(os.pardir)

import torch
from torch import nn
from utils.pyg_graph_models import GraphAttentionPooling, ResNetBlock, BipartiteGCN
from torch_geometric.data import Data
from utils.utils import merge_graphs
from torch_geometric.utils import to_dense_batch
from torch.distributions import Categorical

class BipartiteData(Data):
    def __init__(self, edge_index, x_src, x_dst):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index
        self.x_s = x_src
        self.x_t = x_dst

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super(BipartiteData, self).__inc__(key, value)

def matrix2graph(sparse_matrix):
    subset_num = sparse_matrix.size(0)
    element_num = sparse_matrix.size(1)
    row, col, val = sparse_matrix.coo()
    # prepare edge indices
    edge_indices = torch.stack((row, col), dim=0)
    # prepare subset tensor
    x = torch.zeros(subset_num, 2)
    x += torch.tensor([1, 0])
    # prepare element tensor
    y = torch.zeros(element_num, 3)

    for i in range(len(row)):
        y[col[i].item()] = torch.tensor([0,1,0])
        y[col[i].item()][2] = val[i]

    return BipartiteData(edge_indices, x, y)

class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        node_feature_dim,
        node_output_size,
        batch_norm,
        one_hot_degree,
        num_layers=10
    ):
        super(GraphEncoder, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.node_output_size = node_output_size
        self.one_hot_degree = one_hot_degree
        self.batch_norm = batch_norm
        self.num_layers = num_layers

        one_hot_dim = self.one_hot_degree + 1 if self.one_hot_degree > 0 else 0
        self.bipartite_gcn = BipartiteGCN((self.node_feature_dim - 1, self.node_feature_dim),
                                self.node_output_size, num_layers = self.num_layers, batch_norm = self.batch_norm)
        self.subset_att = GraphAttentionPooling(self.node_output_size)
        self.element_att = GraphAttentionPooling(self.node_output_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, matrixes):
        # construct graph batches
        graphs = [matrix2graph(matrix) for matrix in matrixes]
        x_s, x_e, edge_index, s_batch, e_batch = merge_graphs(graphs, self.device)

        # forward pass
        (batched_subset_feat, batched_element_feat) = self.bipartite_gcn((x_s, x_e), edge_index)
        element_feat_reshape, _ = to_dense_batch(batched_element_feat, e_batch)
        element_feat = self.element_att(batched_element_feat, e_batch)
        element_state_feat = torch.cat(
            (element_feat_reshape, element_feat.unsqueeze(1).expand(-1, element_feat_reshape.shape[1], -1)), dim=-1)

        subset_feat_reshape, _ = to_dense_batch(batched_subset_feat, s_batch)
        subset_feat = self.subset_att(batched_subset_feat, s_batch)
        subset_state_feat = torch.cat(
            (subset_feat_reshape, subset_feat.unsqueeze(1).expand(-1, subset_feat_reshape.shape[1], -1)), dim=-1)

        return element_state_feat, subset_state_feat

class ActorNet(torch.nn.Module):
    def __init__(
        self,
        state_feature_size,
        batch_norm
    ):
        super(ActorNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm

        self.act1_resnet = ResNetBlock(self.state_feature_size, 1, batch_norm=self.batch_norm)
        self.act2_resnet = ResNetBlock(self.state_feature_size * 2, 1, batch_norm=self.batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, element_feat, subset_feat, edge_candidates, known_action=None):
        return self._act(element_feat, subset_feat, edge_candidates, known_action)

    def _act(self, element_feat, subset_feat, edge_candidates, known_action=None):
        if known_action is None:
            known_action = (None, None)
        # roll out 2 acts
        mask1, ready_nodes1 = self._get_mask1(subset_feat.shape[0], subset_feat.shape[1], edge_candidates)
        act1, log_prob1, entropy1 = self._select_node(subset_feat, element_feat, mask1, known_action[0])
        mask2, ready_nodes2 = self._get_mask2(element_feat.shape[0], element_feat.shape[1], edge_candidates, act1)
        act2, log_prob2, entropy2 = self._select_node(subset_feat, element_feat, mask2, known_action[1], act1)
        return torch.stack((act1, act2)), torch.stack((log_prob1, log_prob2)), entropy1 + entropy2

    def _select_node(self, subset_feat, element_feat, mask, known_cur_act=None, prev_act=None, greedy_sel_num=0):
        # neural net prediction
        if prev_act is None: # for act 1
            act_scores = self.act1_resnet(subset_feat).squeeze(-1)
        else: # for act 2
            prev_node_feat = subset_feat[torch.arange(len(prev_act)), prev_act, :]
            state_feat = torch.cat(
                (element_feat, prev_node_feat.unsqueeze(1).expand(-1, element_feat.shape[1], -1)), dim=-1)
            act_scores = self.act2_resnet(state_feat).squeeze(-1)

        # select action
        if greedy_sel_num > 0:
            act_probs = nn.functional.softmax(act_scores + mask, dim=1)
            argsort_prob = torch.argsort(act_probs, dim=-1, descending=True)
            acts = argsort_prob[:, :greedy_sel_num]
            return acts, act_probs[torch.arange(acts.shape[0]).unsqueeze(-1), acts]
        else:
            act_log_probs = nn.functional.log_softmax(act_scores + mask, dim=1)
            dist = Categorical(logits=act_log_probs)
            if known_cur_act is None:
                act = dist.sample()
                return act, dist.log_prob(act), dist.entropy()
            else:
                return known_cur_act, dist.log_prob(known_cur_act), dist.entropy()

    def _get_mask1(self, batch_size, num_nodes, edge_candidates):
        masks = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        ready_nodes = []
        for b in range(batch_size):
            ready_nodes.append([])
            for node, candidates in edge_candidates[b].items():
                if len(candidates) == 0:
                    pass
                else:
                    masks[b, node] = 0
                    ready_nodes[b].append(node)
        return masks, ready_nodes

    def _get_mask2(self, batch_size, num_nodes, edge_candidates, act1):
        masks = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        ready_nodes = []
        for b in range(batch_size):
            ready_nodes.append([])
            candidates = edge_candidates[b][act1[b].item()]
            for index in candidates:
                masks[b, index] = 0.0
                ready_nodes[b].append(index)
        return masks, ready_nodes

class CriticNet(torch.nn.Module):
    def __init__(
        self,
        state_feature_size,
        batch_norm
    ):
        super(CriticNet, self).__init__()
        self.state_feature_size = state_feature_size
        self.batch_norm = batch_norm

        self.critic_resnet = ResNetBlock(self.state_feature_size, 1, batch_norm=self.batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, element_feat, subset_feat):
        return self._eval(element_feat, subset_feat)

    def _eval(self, element_feat, subset_feat):
        # get global features
        element_feat = torch.max(element_feat, dim=1).values
        subset_feat = torch.max(subset_feat, dim=1).values
        state_feat = torch.cat((element_feat, subset_feat), dim=-1)
        state_value = self.critic_resnet(state_feat).squeeze(-1)
        return state_value