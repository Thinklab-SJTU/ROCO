import argparse
import numpy as np
import sys
import time
import random
from ortools.linear_solver import pywraplp
import gurobipy as gp
from mc_ppo_pytorch import parse_arguments
from mc_env import MCEnv
import torch

def evaluate(sparse_matrix):
    row, col, val = sparse_matrix.coo()
    subset_num = sparse_matrix.size(0)
    element_num = sparse_matrix.size(1)
    weights = [0] * element_num

    subset_dict = {x:set() for x in range(subset_num)}
    for i in range(len(row)):
        subset_dict[row[i].item()].add(col[i].item())
        weights[col[i].item()] = val[i].item()
    return subset_num, element_num, weights, subset_dict

def gurobi_search(sparse_matrix, limit, time_limit, MIPFoucus, Heuristics):
    subset_num, element_num, weight_array, subset_dict = evaluate(sparse_matrix)
    m = gp.Model("MC")

    # Initialize variables
    X = m.addVars(subset_num, vtype=gp.GRB.BINARY, name="X")
    Y = m.addVars(element_num, vtype=gp.GRB.BINARY, name="Y")
    m.update()

    # Set Objective
    m.setObjective(sum([Y[i] * weight_array[i] for i in range(element_num)]), gp.GRB.MAXIMIZE)

    # Add Constraints
    m.addConstr(sum([X[i] for i in range(subset_num)]) <= limit)
    for j in range(element_num):
        m.addConstr(sum([(1 if j in subset_dict[i] else 0) * X[i] for i in range(subset_num)]) >= Y[j])

    m.Params.LogToConsole = False
    m.Params.TimeLimit = time_limit
    m.Params.MIPFocus = MIPFoucus
    m.Params.Heuristics = Heuristics

    prev_time = time.time()
    m.optimize()
    comp_time = time.time() - prev_time

    # if m.status == gp.GRB.Status.OPTIMAL:
    #     print(f"gurobi has found the optimal solution.")
    # else:
    #     print(f"gurobi failed to find the optimal solution within the time limit.")
    # print(m.status)
    obj = m.objVal
    if obj < 0:
        obj = 0
    S = []
    # for k, v in m.getAttr('X', X).items():
    #     if v == 1:
    #         S.append(k)

    return obj, comp_time, S

def update_time_ratio(mc_env):
    sparse_matrix, limit = mc_env.baseline_graph
    cur_time_list = []
    for _ in range(9):
        obj, cur_time, S = mc_env.solve_feasible_mc(sparse_matrix, limit, mc_env.baseline_time * 100, 'gurobi')
        cur_time_list.append(cur_time)
    cur_time_list.sort()
    cur_time = np.mean(cur_time_list[2:7])
    mc_env.time_limit_ratio = cur_time / mc_env.baseline_time

    return mc_env.time_limit_ratio

def grid_search_search_best_para(args, mc_env, tuples_test):
    record = dict()

    for MIPFoucus in range(4):
        for Heuristics in [0, 0.05, 0.10]:
            cur_name = f'MIPFoucus={MIPFoucus}_Heuristics={Heuristics}'
            cur_obj_list = []
            time_limit_ratio = update_time_ratio(mc_env)
            print(f"----------{cur_name}_time_limit_ratio={time_limit_ratio}----------")
            time_limit = time_limit_ratio * args.time_limit
            for graph_index, (inp_matrix, limit, edge_candidates, ori_greedy, baselines, _) in enumerate(tuples_test):
                obj, comp_time, S = gurobi_search(inp_matrix, limit, time_limit, MIPFoucus, Heuristics)
                cur_obj_list.append(obj)
                print(f"{graph_index}: obj={obj}, time={comp_time}")
            record[cur_name] = np.mean(cur_obj_list)

    for key in record.keys():
        print(key, record[key])

if __name__ == '__main__':
    args = parse_arguments()

    # create environment
    mc_env = MCEnv(args.solver_type, args.subset_size, args.element_size, args.time_limit)
    args.node_feature_dim = 3

    # get current device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load training/testing data
    tuples_train, tuples_test = mc_env.generate_tuples(args.train_sample, args.test_sample, 0)

    grid_search_search_best_para(args, mc_env, tuples_test)


