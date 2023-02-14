import numpy as np
import random
import torch
import os
import time
from copy import deepcopy
from torch_sparse import SparseTensor
from mc_algorithms import greedy_search_naive, ortools_search, gurobi_search

class MCEnv(object):
    def __init__(self, solver_type='greedy_naive', subset_size=500, element_size=1000, time_limit=1.):
        self.subset_size = subset_size
        self.element_size = element_size
        self.solver_type = solver_type
        self.time_limit = time_limit
        self.time_limit_ratio = 1
        self.steps = 0
        self.process_dataset()
        # self.available_solvers = ('greedy_naive', 'gurobi', 'scip', 'cbc')
        self.available_solvers = [solver_type]
        assert solver_type in self.available_solvers, "Unknown solver type!"

    def process_dataset(self):
        print("\nPreparing dataset.\n")
        dirpath = f'mc_data/mc_{self.subset_size}_{self.element_size}'
        mcfiles = []
        for i in range(1, 71):
            file = f'mc_{self.subset_size}_{self.element_size}_{i}.txt'
            mcfiles.append(dirpath + '/' + file)
        self.num_edges = []
        self.graphs = [self.construct_sparse_matrix(gf) for gf in mcfiles]
        print(np.mean(self.num_edges))
        # Time_baseline_graph
        self.baseline_time = 0.533
        self.baseline_graph = self.construct_sparse_matrix(f"mc_data/mc_baseline_{self.baseline_time}.txt")


    def update_time_ratio(self):
        sparse_matrix, limit = self.baseline_graph
        cur_time_list = []
        for _ in range(9):
            obj, cur_time, S = self.solve_feasible_mc(sparse_matrix, limit, self.baseline_time * 100, 'gurobi')
            cur_time_list.append(cur_time)
        cur_time_list.sort()
        cur_time = np.mean(cur_time_list[2:7])
        self.time_limit_ratio = cur_time / self.baseline_time

    def construct_sparse_matrix(self, graph_file):
        with open(graph_file,'r') as data:
            subset_num, element_num, limit = next(data).strip().split(',')
            row = []; col = []; val = []
            for line in data:
                _, element_id, subset_id, weight = line.strip().split(',')
                row.append(int(subset_id)); col.append(int(element_id)); val.append(int(weight))
        self.num_edges.append(len(row))
        sizes = (int(subset_num), int(element_num))
        return (SparseTensor(torch.tensor(row), None, torch.tensor(col), torch.tensor(val), sizes), int(limit))

    def generate_tuples(self, num_train_samples, num_test_samples, rand_id):
        if self.solver_type != 'greedy_naive':
            self.update_time_ratio()

        random.seed(int(rand_id))
        np.random.seed(int(rand_id + num_train_samples))

        training_tuples = []
        testing_tuples = []

        return_tuples = training_tuples
        sum_num_nodes = 0
        sum_num_edges = 0
        for i, (sparse_matrix, limit) in enumerate(self.graphs):
            sum_num_nodes += (sparse_matrix.size(0) + sparse_matrix.size(1))
            mc_solutions = {}
            mc_times = {}
            for key in self.available_solvers:
                obj, sec, S = self.solve_feasible_mc(sparse_matrix, limit, self.time_limit_ratio * self.time_limit, key)
                mc_solutions[key] = obj
                mc_times[key] = sec
                if key == self.solver_type:
                    edge_candidates = self.get_attack_edge_candidates(sparse_matrix, S)

            print(f'id {i}'
                f'{",".join([f"{x} weight={mc_solutions[x]:.1f} time={mc_times[x]:.2f}" for x in self.available_solvers])}')

            return_tuples.append((
                sparse_matrix,
                limit,
                edge_candidates,
                mc_solutions[self.solver_type],
                mc_solutions,
                mc_times,
            ))
            if i == num_train_samples - 1 or i == num_train_samples + num_test_samples - 1:
                print(f'average number of nodes: {sum_num_nodes / len(return_tuples)}')
                print(f'average number of edges: {sum_num_edges / len(return_tuples)}')
                sum_num_nodes = 0
                for solver_name in self.available_solvers:
                    print(f'{solver_name} average amount='
                        f'{torch.mean(torch.tensor([tup[4][solver_name] for tup in return_tuples], dtype=torch.float)):.1f}')
                return_tuples = testing_tuples
            if i == num_train_samples + num_test_samples - 1:
                break
        return training_tuples, testing_tuples

    def step(self, sparse_matrix, limit, act, prev_solution):
        if self.steps % 100 == 0 and self.solver_type != 'greedy_naive':
            self.update_time_ratio()
        self.steps += 1

        row, col, val = sparse_matrix.coo()
        new_row = deepcopy(row)
        new_col = deepcopy(col)
        new_val = deepcopy(val)
        new_sizes = (sparse_matrix.size(0), sparse_matrix.size(1))
        # add additional edges
        for i in range(len(row)):
            if col[i] == act[1]:
                target_val = val[i]
                break
        new_row = torch.cat((new_row, act[0].unsqueeze(-1)))
        new_col = torch.cat((new_col, act[1].unsqueeze(-1)))
        new_val = torch.cat((new_val, target_val.unsqueeze(-1)))

        new_sparse_matrix = SparseTensor(new_row, None, new_col, new_val, new_sizes)
        new_solution, _, new_S = self.solve_feasible_mc(new_sparse_matrix, limit, self.time_limit_ratio * self.time_limit, self.solver_type)
        new_edge_candidate = self.get_attack_edge_candidates(new_sparse_matrix, new_S)
        reward = prev_solution - new_solution

        done = all([len(x) == 0 for x in new_edge_candidate.values()])

        return reward, new_sparse_matrix, new_edge_candidate, new_solution, done

    @staticmethod
    def get_attack_edge_candidates(sparse_matrix, subsets):
        subset_num = sparse_matrix.size(0)
        element_num = sparse_matrix.size(1)
        row, col, val = sparse_matrix.coo()
        edge_candidate = {x: set() for x in range(subset_num)}
        elements = set()
        for i in range(len(row)):
            elements.add(col[i].item())
            edge_candidate[row[i].item()].add(col[i].item())
        for i in range(subset_num):
            if i in subsets:
                edge_candidate[i].clear()
            else:
                edge_candidate[i] = elements - edge_candidate[i]
        return edge_candidate

    def solve_feasible_mc(self, sparse_matrix, limit, time_limit, key):
        # prev_time = time.time()
        if key == 'greedy_naive':
            obj, comp_time, S = greedy_search_naive(sparse_matrix, limit)
        elif key == 'cbc':
            obj, comp_time, S = ortools_search(sparse_matrix, limit, 'cbc', time_limit)
        elif key == 'scip':
            obj, comp_time, S = ortools_search(sparse_matrix, limit, 'scip', time_limit)
        elif key == 'gurobi':
            # obj, S = ortools_search(sparse_matrix, limit, 'gurobi', time_limit)
            obj, comp_time, S = gurobi_search(sparse_matrix, limit, time_limit)
        else:
            raise ValueError("Unknown solver type!")
        # comp_time = time.time() - prev_time
        return obj, comp_time, S

