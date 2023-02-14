from torch_sparse import SparseTensor
import gurobipy as gp
from ortools.linear_solver import pywraplp

def evaluate(sparse_matrix):
    row, col, val = sparse_matrix.coo()
    strategy_num = sparse_matrix.size(0)
    events_dict = {x:{'black':set(),'white':set()} for x in range(strategy_num)}
    black_dict = {}
    for i in range(len(row)):
        if val[i] > 0:
            events_dict[row[i].item()]['black'].add(col[i].item())
            black_dict[col[i].item()] = val[i].item()
        else:
            events_dict[row[i].item()]['white'].add(col[i].item())
    return events_dict, black_dict, strategy_num

def evaluateGurobi(sparse_matrix):
    row, col, val = sparse_matrix.coo()
    strategy_num = sparse_matrix.size(0)
    events_num = sparse_matrix.size(1)
    money = [0] *  events_num
    black_events = set()
    white_events = set()
    events_dict = {x:set() for x in range(strategy_num)}
    for i in range(len(row)):
        if val[i] > 0:
            black_events.add(col[i].item())
            money[col[i].item()] = val[i].item()
            events_dict[row[i].item()].add(col[i].item())
        else:
            white_events.add(col[i].item())
            events_dict[row[i].item()].add(col[i].item())
    return strategy_num, events_num, money, events_dict, black_events, white_events

def ortools_search(sparse_matrix, limit):
    strategy_num, events_num, money, events_dict, black_events, white_events = evaluateGurobi(sparse_matrix)
    # create the mip solver with the Gurobi backend
    solver = pywraplp.Solver.CreateSolver('gurobi')

    # Initialize variables
    X = {}
    Y = {}
    for i in range(strategy_num):
        X[i] = solver.IntVar(0, 1, f'X[{i}]')
    for i in range(events_num):
        Y[i] = solver.IntVar(0, 1, f'Y[{i}]')

    # Add Constraints
    constraint_expr1 = [Y[i] for i in white_events]
    solver.Add(sum(constraint_expr1) <= limit)
    for j in range(events_num):
        first = Y[j] - 0.5
        second = 0.5 - sum([(1 if j in events_dict[i] else 0) * X[i] for i in range(strategy_num)])
        solver.Add(first * second <= 0)

    # Set Objective
    obj_expr = [Y[i] * money[i] for i in black_events]
    solver.Maximize(solver.Sum(obj_expr))

    # Set time limit 0.5s
    solver.SetTimeLimit(0.5 * 1000)

    status = solver.Solve()

    obj = solver.Objective().Value()
    S = []
    for i in range(strategy_num):
        if X[i].solution_value() == 1:
            S.append(i)

    return obj, S

def gurobi_search(sparse_matrix, limit):
    strategy_num, events_num, money, events_dict, black_events, white_events = evaluateGurobi(sparse_matrix)
    m = gp.Model("Fraud")
    X = m.addVars(strategy_num, vtype=gp.GRB.BINARY, name="X")
    Y = m.addVars(events_num, vtype=gp.GRB.BINARY, name="Y")
    m.update()
    
    m.setObjective(gp.quicksum(Y[i]*money[i] for i in black_events), gp.GRB.MAXIMIZE)
    
    m.addConstr(gp.quicksum(Y[i] for i in white_events) <= limit)
    m.addConstrs((Y[j]-0.5)*(0.5-gp.quicksum((1 if j in events_dict[i] else 0) * X[i] for i in range(strategy_num))) <= 0 for j in range(events_num))
    
    m.Params.LogToConsole=False
    m.Params.TimeLimit = 1.0
    m.optimize()
    
    obj = m.objVal
    S = []
    for k, v in m.getAttr('X', X).items():
        if v == 1:
            S.append(k)
    
    # print(S)
    return obj, S

def local_search(sparse_matrix, limit):
    events_dict, black_dict, strategy_num = evaluate(sparse_matrix)
    S = []
    V = list(range(strategy_num))
    black_set = set()
    white_set = set()
    obj = 0; con = 0
    while True:
        flag = False
        for v in V:
            S_ = S + [v]
            new_black = events_dict[v]['black'] - black_set
            new_white = events_dict[v]['white'] - white_set
            con_incr = len(new_white)
            if len(new_black) > 0 and con + con_incr <= limit:
                S = S_
                V.remove(v)
                flag = True
                black_set = black_set | events_dict[v]['black']
                white_set = white_set | events_dict[v]['white']
                for i in new_black:
                    obj += black_dict[i]
                con += con_incr
                break
        if flag:
            continue
        else:
            break
    return obj, S

def greedy_search_naive(sparse_matrix, limit):
    events_dict, black_dict, strategy_num = evaluate(sparse_matrix)
    S = []
    V = list(range(strategy_num))
    black_set = set()
    white_set = set()
    obj = 0; con = 0
    while True:
        best_incr = 0
        best_v = None
        best_con_incr = 0
        for v in V:
            obj_incr = 0
            new_black = events_dict[v]['black'] - black_set
            new_white = events_dict[v]['white'] - white_set
            for i in new_black:
                obj_incr += black_dict[i]
            con_incr = len(new_white)
            if obj_incr >= best_incr and con + con_incr <= limit:
                best_incr = obj_incr
                best_con_incr = con_incr
                best_v = v
        if best_v is not None:
            S.append(best_v)
            V.remove(best_v)
            black_set = black_set | events_dict[best_v]['black']
            white_set = white_set | events_dict[best_v]['white']
            obj += best_incr
            con += best_con_incr
        else:
            break

    return obj, S

def greedy_average_search(sparse_matrix, limit):
    events_dict, black_dict, strategy_num = evaluate(sparse_matrix)
    S = []
    V = list(range(strategy_num))
    black_set = set()
    white_set = set()
    eps = 1e-5
    obj = 0; con = 0
    while True:
        best_incr = 0
        best_con_inr = 0
        best_v = None
        for v in V:
            obj_incr = 0
            new_black = events_dict[v]['black'] - black_set
            new_white = events_dict[v]['white'] - white_set
            for i in new_black:
                obj_incr += black_dict[i]
            con_incr = len(new_white)
            if obj_incr / (con_incr + eps) >= best_incr and con + con_incr <= limit:
                best_incr = obj_incr / (con_incr + eps)
                best_con_inr = con_incr
                best_v = v
        if best_v is not None:
            S.append(best_v)
            V.remove(best_v)
            con += best_con_inr
            obj += best_incr * (best_con_inr + eps)
            black_set = black_set | events_dict[best_v]['black']
            white_set = white_set | events_dict[best_v]['white']
        else:
            break
    
    return obj, S