import functools
import networkx as nx
import numpy as np
import random
import torch
import os
import time
from torch_geometric.data import Data, DataLoader
from fc_algorithms import greedy_search_naive, greedy_average_search, local_search, gurobi_search
from copy import deepcopy
from torch_sparse import SparseTensor

class FCEnv(object):
    def __init__(self, solver_type='greedy_naive', strategy_size=30, event_size=3000, is_attack=False, modify_nodes=False):
        self.strategy_size = strategy_size
        self.event_size = event_size
        self.solver_type = solver_type
        self.process_dataset()
        # self.available_solvers = ('greedy_naive', 'greedy_average', 'local', 'gurobi')
        self.available_solvers = (solver_type,)
        self.is_attack = is_attack
        self.modify_nodes = modify_nodes
        assert solver_type in self.available_solvers
    
    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")
        dirpath = f'fraud_data/{self.strategy_size}_{self.event_size}'
        fcfiles = []
        self.num_edges = []

        for file in os.listdir(dirpath):
            # optionally add files
            _, strategy_num, event_num, _, _ = file.split('_')
            if int(strategy_num) == self.strategy_size and int(event_num) == self.event_size:
                fcfiles.append(dirpath+'/'+file)
        self.graphs = [self.construct_sparse_matrix(gf) for gf in fcfiles]
        print(np.mean(self.num_edges))
        
    def construct_sparse_matrix(self, graph_file):
        with open(graph_file,'r') as data:
            strategy_num, events_num, limit = next(data).strip().split(',')
            row = []; col = []; val = []
            for line in data:
                _, event_id, _, strategy_id, amt, flag = line.strip().split(',')
                if float(flag) > 0:
                    row.append(int(strategy_id)); col.append(int(event_id)); val.append(float(amt))
                else:
                    row.append(int(strategy_id)); col.append(int(event_id)); val.append(-float(amt))
        self.num_edges.append(len(row))
        sizes = (int(strategy_num), int(events_num))
        return (SparseTensor(torch.tensor(row), None, torch.tensor(col), torch.tensor(val), sizes), int(limit))
        
    def construct_matrix(self, graph_file):
        with open(graph_file,'r') as data:
            strategy_num, events_num, limit = next(data).strip().split(',')
            matrix = torch.zeros((int(strategy_num),int(events_num)))
            for line in data:
                _, event_id, _, strategy_id, amt, flag = line.strip().split(',')
                if float(flag) > 0:
                    matrix[int(strategy_id)][int(event_id)] = float(amt)
                else:
                    matrix[int(strategy_id)][int(event_id)] = -float(amt)
        
        return (matrix, int(limit))
    
    def generate_tuples(self, num_train_samples, num_test_samples, rand_id):
        random.seed(int(rand_id))
        np.random.seed(int(rand_id + num_train_samples))
        
        training_tuples = []
        testing_tuples = []
        
        return_tuples = training_tuples
        sum_num_nodes = 0
        for i, (sparse_matrix, limit) in enumerate(self.graphs):
            # edge_candidates = self.get_edge_candidates(matrix)
            sum_num_nodes += (sparse_matrix.size(0) + sparse_matrix.size(1))
            
            fc_solutions = {}
            fc_times = {}
            for key in self.available_solvers:
                obj, sec, S = self.solve_feasible_fc(sparse_matrix, limit, key)
                fc_solutions[key] = obj
                fc_times[key] = sec
                if key == self.solver_type:
                    if self.is_attack and not self.modify_nodes:
                        edge_candidates = self.get_attack_edge_candidates(sparse_matrix, S)
                    elif self.is_attack and self.modify_nodes:
                        edge_candidates = self.get_node_candidates(sparse_matrix)
                    else:
                        edge_candidates = self.get_edge_candidates(sparse_matrix, S)
            print(f'id {i}'
                f'{";".join([f"{x} amount={fc_solutions[x]:.2f} time={fc_times[x]:.2f}" for x in self.available_solvers])}')
            return_tuples.append((
                sparse_matrix,
                limit,
                edge_candidates,
                fc_solutions[self.solver_type],
                fc_solutions,
                fc_times,
            ))
            if i == num_train_samples - 1 or i == num_train_samples + num_test_samples - 1:
                print(f'average number of nodes: {sum_num_nodes / len(return_tuples)}')
                sum_num_nodes = 0
                for solver_name in self.available_solvers:
                    print(f'{solver_name} average amount='
                        f'{torch.mean(torch.tensor([tup[4][solver_name] for tup in return_tuples], dtype=torch.float)):.4f}')
                return_tuples = testing_tuples
            if i == num_train_samples + num_test_samples - 1:
                break
        return training_tuples, testing_tuples

    def step(self, sparse_matrix, limit, act, prev_solution, defense = False):
        if self.is_attack and not self.modify_nodes and not defense:
            return self.step_attack(sparse_matrix, limit, act, prev_solution)
        if self.is_attack and self.modify_nodes and not defense:
            return self.step_node_attack(sparse_matrix, limit, act, prev_solution)
        
        if isinstance(act, torch.Tensor):
            act = (act[0].item(), act[1].item())
        row, col, val = sparse_matrix.coo()
        # remove the black edge
        for i in range(len(row)):
            if row[i] == act[0] and col[i] == act[1]:
                break
        new_row = torch.cat((row[:i],row[i+1:]), dim=0)
        new_col = torch.cat((col[:i],col[i+1:]), dim=0)
        new_val = torch.cat((val[:i],val[i+1:]), dim=0)
        new_sizes = (sparse_matrix.size(0), sparse_matrix.size(1))

        new_sparse_matrix = SparseTensor(new_row, None, new_col, new_val, new_sizes)
        new_solution, _, new_S = self.solve_feasible_fc(new_sparse_matrix, limit, self.solver_type)
        new_edge_candidate = self.get_edge_candidates(new_sparse_matrix, new_S)
        reward = new_solution - prev_solution
        done = all([len(x) == 0 for x in new_edge_candidate.values()])
        
        return reward, new_sparse_matrix, new_edge_candidate, new_solution, done
    
    def step_attack(self, sparse_matrix, limit, act, prev_solution):
        row, col, val = sparse_matrix.coo()
        new_row = deepcopy(row)
        new_col = deepcopy(col)
        new_val = deepcopy(val)
        new_sizes = (sparse_matrix.size(0), sparse_matrix.size(1))
        # add additional black edges
        for i in range(len(row)):
            if col[i] == act[1]:
                target_val = val[i]
                break
        new_row = torch.cat((new_row, act[0].unsqueeze(-1)))
        new_col = torch.cat((new_col, act[1].unsqueeze(-1)))
        new_val = torch.cat((new_val, target_val.unsqueeze(-1)))
        
        new_sparse_matrix = SparseTensor(new_row, None, new_col, new_val, new_sizes)
        new_solution, _, new_S = self.solve_feasible_fc(new_sparse_matrix, limit, self.solver_type)
        new_edge_candidate = self.get_attack_edge_candidates(new_sparse_matrix, new_S)
        reward = prev_solution - new_solution

        done = all([len(x) == 0 for x in new_edge_candidate.values()])
        
        return reward, new_sparse_matrix, new_edge_candidate, new_solution, done
    
    # the attack method and node candidates
    def step_node_attack(self, sparse_matrix, limit, act, prev_solution):
        if isinstance(act, torch.Tensor):
            act = act.item()
        row, col, val = sparse_matrix.coo()
        new_row = deepcopy(row)
        new_col = deepcopy(col)
        new_val = deepcopy(val)
        new_sizes = (sparse_matrix.size(0), sparse_matrix.size(1))
        # change the white event to black
        for i in range(len(row)):
            if col[i] == act:
                new_val[i] = -val[i]
        # Will the memory be shared or not?
        new_sparse_matrix = SparseTensor(new_row, None, new_col, new_val, new_sizes)
        new_solution, _, _ = self.solve_feasible_fc(new_sparse_matrix, limit, self.solver_type)
        new_node_candidates = self.get_node_candidates(new_sparse_matrix)
        reward = prev_solution - new_solution
        done = (len(new_node_candidates) == 0)
        
        return reward, new_sparse_matrix, new_node_candidates, new_solution, done
    
    @staticmethod
    def get_edge_candidates(sparse_matrix, strategies):
        num_strategy = sparse_matrix.size(0)
        num_events = sparse_matrix.size(1)
        row, col, val = sparse_matrix.coo()
        edge_candidate = {x: set() for x in range(num_strategy)}
        for i in range(len(row)):
            if row[i].item() in strategies and val[i].item() > 0:
                edge_candidate[row[i].item()].add(col[i].item())
            else:
                pass
            
        return edge_candidate
    
    @staticmethod
    def get_attack_edge_candidates(sparse_matrix, strategies):
        num_strategy = sparse_matrix.size(0)
        num_events = sparse_matrix.size(1)
        row, col, val = sparse_matrix.coo()
        edge_candidate = {x: set() for x in range(num_strategy)}
        black_events = set()
        for i in range(len(row)):
            if val[i].item() > 0:
                black_events.add(col[i].item())
                edge_candidate[row[i].item()].add(col[i].item())
        for i in range(num_strategy):
            if i in strategies:
                edge_candidate[i].clear()
            else:
                edge_candidate[i] = black_events - edge_candidate[i]
        return edge_candidate

    @staticmethod
    def get_node_candidates(sparse_matrix):
        node_candidates = set()
        row, col, val = sparse_matrix.coo()
        for i in range(len(row)):
            if val[i].item() < 0:
                node_candidates.add(col[i].item())
            else:
                pass
        return node_candidates
    
    def solve_feasible_fc(self, sparse_matrix, limit, key):
        prev_time = time.time()
        if key == 'greedy_naive':
            obj, S = greedy_search_naive(sparse_matrix, limit)
        elif key == 'greedy_average':
            obj, S = greedy_average_search(sparse_matrix, limit)
        elif key == 'local':
            obj, S = local_search(sparse_matrix, limit)
        elif key == 'gurobi':
            obj, S = gurobi_search(sparse_matrix, limit)
        else:
            raise ValueError(f'{self.solver_type} is not implemented.')
        comp_time = time.time() - prev_time
        return obj, comp_time, S