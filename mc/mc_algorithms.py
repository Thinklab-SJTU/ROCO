from torch_sparse import SparseTensor
from ortools.linear_solver import pywraplp
import gurobipy as gp
import time

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

def greedy_search_naive(sparse_matrix, limit):
    prev_time = time.time()
    subset_num, _, weight_array, subset_dict = evaluate(sparse_matrix)
    S = []
    V = list(range(subset_num))
    element_set = set()
    obj = 0
    for _ in range(limit):
        best_incr = 0
        best_v = None
        for v in V:
            obj_incr = 0
            new_elements = subset_dict[v] - element_set
            for i in new_elements:
                obj_incr += weight_array[i]
            if obj_incr >= best_incr:
                best_incr = obj_incr
                best_v = v
        if best_v is not None:
            S.append(best_v)
            V.remove(best_v)
            element_set = element_set | subset_dict[best_v]
            obj += best_incr
        else:
            break
    comp_time = time.time() - prev_time
    return obj, comp_time, S

def gurobi_search(sparse_matrix, limit, time_limit):
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

    prev_time = time.time()
    m.optimize()
    comp_time = time.time() - prev_time

    # if m.status == gp.GRB.Status.OPTIMAL:
    #     print(f"gurobi has found the optimal solution.")
    # else:
    #     print(f"gurobi failed to find the optimal solution within the time limit.")

    obj = m.objVal
    S = []
    for k, v in m.getAttr('X', X).items():
        if v == 1:
            S.append(k)

    return obj, comp_time, S

def ortools_search(sparse_matrix, limit, solver_name, time_limit):
    subset_num, element_num, weight_array, subset_dict = evaluate(sparse_matrix)
    # create the mip solver with the Gurobi backend
    if solver_name == 'gurobi':
        solver = pywraplp.Solver('SolveMC', pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING)
    elif solver_name == 'scip':
        solver = pywraplp.Solver('SolveMC', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    elif solver_name == 'cbc':
        solver = pywraplp.Solver('SolveMC', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    else:
        raise ValueError("Unrecognized solver.")

    # Initialize variables
    X = [0] * subset_num
    Y = [0] * element_num
    for i in range(subset_num):
        X[i] = solver.IntVar(0, 1, f'X[{i}]')
    for i in range(element_num):
        Y[i] = solver.IntVar(0, 1, f'Y[{i}]')

    # Add Constraints
    solver.Add(sum(X) <= limit)
    for j in range(element_num):
        sum_j = sum([(1 if j in subset_dict[i] else 0) * X[i] for i in range(subset_num)])
        solver.Add(sum_j >= Y[j])

    # Set Objective
    obj_expr = [Y[i] * weight_array[i] for i in range(element_num)]
    solver.Maximize(solver.Sum(obj_expr))

    # Set time limit
    solver.SetTimeLimit(int(time_limit * 1000))

    prev_time = time.time()
    status = solver.Solve()
    comp_time = time.time() - prev_time
    # if status == pywraplp.Solver.OPTIMAL:
    #     print(f"{solver_name} has found the optimal solution.")
    # else:
    #     print(f"{solver_name} failed to find the optimal solution within the time limit.")

    obj = solver.Objective().Value()
    S = []
    for i in range(subset_num):
        if X[i].solution_value() == 1:
            S.append(i)

    # print(S)
    return obj, comp_time, S