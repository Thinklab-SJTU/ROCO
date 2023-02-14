import time
import itertools
import torch
from copy import deepcopy
from torch_sparse import SparseTensor
from fc_algorithms import greedy_search_naive, greedy_average_search, local_search, gurobi_search
import numpy as np

def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))

def beam_search_step_kernel(idx, act_n_sel,
                            acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                            graph_list, limit, act_list, prob_list, orig_greedy, fc_env, defense):
    beam_idx = idx // act_n_sel ** 2
    act1_idx = idx // act_n_sel % act_n_sel
    act2_idx = idx % act_n_sel
    act1, prob1 = acts1[beam_idx, act1_idx], probs1[beam_idx, act1_idx].item()
    act2, prob2 = acts2[beam_idx, act1_idx, act2_idx], probs2[beam_idx, act1_idx, act2_idx].item()
    ready_nodes_1 = ready_nodes1[beam_idx]
    ready_nodes_2 = ready_nodes2_flat[beam_idx * act_n_sel + act1_idx]
    
    if act1 in ready_nodes_1 and act2 in ready_nodes_2:
        assert prob1 > 0
        assert prob2 > 0
        reward, new_matrix, edge_candidates, new_greedy, done = \
            fc_env.step(graph_list[beam_idx], limit, (act1, act2), orig_greedy, defense)
        return (
            new_matrix,
            edge_candidates,
            reward,
            act_list[beam_idx] + [(act1, act2)],
            prob_list[beam_idx] + [(prob1, prob2)],
            done
        )
    else:
        return None

def node_beam_search_step_kernel(idx, act_n_sel, 
                                acts, probs, ready_nodes,
                                matrix_list, limit, act_list, prob_list, orig_greedy, fc_env, defense):
    beam_idx = idx // act_n_sel
    act_idx = idx % act_n_sel
    act, prob = acts[beam_idx, act_idx].item(), probs[beam_idx, act_idx].item()
    ready_nodes = ready_nodes[beam_idx]
    
    if act in ready_nodes:
        assert prob > 0
        reward, new_matrix, node_candidates, new_greedy, done = \
            fc_env.step(matrix_list[beam_idx], limit, act, orig_greedy, defense)
        return (
            new_matrix,
            node_candidates,
            reward,
            act_list[beam_idx] + [act],
            prob_list[beam_idx] + [prob],
            done
        )
    else:
        return None
    
def beam_search(policy_model, fc_env, inp_matrix, limit, edge_candidates, greedy_cost, max_actions, beam_size=5, multiprocess_pool=None, args=None, defense=False):
    start_time = time.time()

    state_encoder = policy_model.state_encoder
    actor_net = policy_model.actor_net

    orig_greedy = greedy_cost
    best_tuple = (
        deepcopy(inp_matrix),  # input lower-left adjacency matrix
        edge_candidates,  # edge candidates
        0,  # accumulated reward
        [],  # actions
        [],  # probabilities
        False,
    )
    topk_graphs = [best_tuple]
    # TODO
    beam_size = 3
    act_n_sel = beam_size
    best_reward_each_step = np.zeros(max_actions + 1)
    for step in range(max_actions):
        print(step)
        matrix_list, edge_cand_list, reward_list, act_list, prob_list = [], [], [], [], []
        for matrix, edge_cand, reward, acts, probs, done in topk_graphs:
            if done:
                continue
            matrix_list.append(matrix)
            edge_cand_list.append(edge_cand)
            reward_list.append(reward)
            act_list.append(acts)
            prob_list.append(probs)

        event_state_feat, strategy_state_feat = state_encoder(matrix_list)
        if args.modify_nodes:
            mask, ready_nodes = actor_net._get_node_mask(event_state_feat.shape[0], event_state_feat.shape[1], edge_cand_list)
            acts, probs = actor_net._select_node(event_state_feat, strategy_state_feat, mask, greedy_sel_num=act_n_sel)
            acts, probs = acts.cpu(), probs.cpu()
        else:
            mask1, ready_nodes1 = actor_net._get_mask1(strategy_state_feat.shape[0], strategy_state_feat.shape[1], edge_cand_list)
            acts1, probs1 = actor_net._select_node(strategy_state_feat, event_state_feat, mask1, greedy_sel_num=act_n_sel)
            acts1_flat, probs1_flat = acts1.reshape(-1), probs1.reshape(-1)
            mask2_flat, ready_nodes2_flat = actor_net._get_mask2(
                event_state_feat.shape[0] * act_n_sel, event_state_feat.shape[1], repeat_interleave(edge_cand_list, act_n_sel),
                acts1_flat)
            acts2_flat, probs2_flat = actor_net._select_node(
                strategy_state_feat.repeat_interleave(act_n_sel, dim=0), event_state_feat.repeat_interleave(act_n_sel, dim=0), mask2_flat, prev_act=acts1_flat, greedy_sel_num=act_n_sel)
            acts2, probs2 = acts2_flat.reshape(-1, act_n_sel, act_n_sel), probs2_flat.reshape(-1, act_n_sel, act_n_sel)
            acts1, acts2, probs1, probs2 = acts1.cpu(), acts2.cpu(), probs1.cpu(), probs2.cpu()

        def kernel_func_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                    matrix_list, limit, act_list, prob_list,
                    orig_greedy, fc_env, defense
                )
        
        def kernel_func_node_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts, probs, ready_nodes,
                    matrix_list, limit, act_list, prob_list,
                    orig_greedy, fc_env, defense
                )
                
        if args.modify_nodes:
            if multiprocess_pool:
                pool_map = multiprocess_pool.starmap_async(
                    node_beam_search_step_kernel, kernel_func_node_feeder(len(matrix_list) * act_n_sel))
                tmp_graphs = pool_map.get()
            else:
                tmp_graphs = [node_beam_search_step_kernel(*x) for x in kernel_func_node_feeder(len(matrix_list) * act_n_sel)]
        else:
            if multiprocess_pool:
                pool_map = multiprocess_pool.starmap_async(
                    beam_search_step_kernel, kernel_func_feeder(len(matrix_list) * act_n_sel ** 2))
                tmp_graphs = pool_map.get()
            else:
                tmp_graphs = [beam_search_step_kernel(*x) for x in kernel_func_feeder(len(matrix_list) * act_n_sel ** 2)]

        searched_graphs = []
        for graph_tuple in tmp_graphs:
            if graph_tuple is not None:
                searched_graphs.append(graph_tuple)
        
        if len(searched_graphs) == 0:
            break

        # find the best action
        searched_graphs.sort(key=lambda x: x[2], reverse=True)

        if searched_graphs[0][2] > best_tuple[2]:
            best_tuple = searched_graphs[0]

        best_reward_each_step[step] = best_tuple[2]

        # find the topk expandable actions
        topk_graphs = searched_graphs[:beam_size]

    best_solution_each_step = orig_greedy - best_reward_each_step
    ret_solution = orig_greedy - best_tuple[2] if (args.is_attack and not defense) else orig_greedy + best_tuple[2]
    return {
        'reward': best_tuple[2],
        'solution': ret_solution,
        'acts': best_tuple[3],
        'probs': best_tuple[4],
        'time': time.time() - start_time,
        'best_reward_each_step': best_reward_each_step,
        'best_solution_each_step': best_solution_each_step
    }

def metric_calculate(fc_env, inp_matrix, limit, bs_result, baselines, args, defense_net = None, mp_pool = None):
    row, col, val = inp_matrix.coo()
    new_row = deepcopy(row)
    new_col = deepcopy(col)
    new_val = deepcopy(val)
    new_sizes = (inp_matrix.size(0), inp_matrix.size(1))
    
    # add additional black edges
    for act in bs_result["acts"]:
        for i in range(len(row)):
            if col[i] == act[1]:
                target_val = val[i]
                break
        new_row = torch.cat((new_row, act[0].unsqueeze(-1)))
        new_col = torch.cat((new_col, act[1].unsqueeze(-1)))
        new_val = torch.cat((new_val, target_val.unsqueeze(-1)))
    
    cur_matrix = SparseTensor(new_row, None, new_col, new_val, new_sizes)
    cur_naive_obj, cur_naive_S = greedy_search_naive(cur_matrix, limit)
    cur_average_obj, cur_average_S = greedy_average_search(cur_matrix, limit)
    cur_local_obj, cur_local_S = local_search(cur_matrix, limit)
    cur_gurobi_obj, cur_gurobi_S = gurobi_search(cur_matrix, limit)
    
    print(f'\t \t'
        f'cur_naive {cur_naive_obj:.4f} \t'
        f'cur_average {cur_average_obj:.4f} \t'
        f'cur_local {cur_local_obj:.4f} \t'
        f'cur_gurobi {cur_gurobi_obj:.4f} \t')
    
    ori_greedy = baselines["greedy_naive"]
    edge_candidates = fc_env.get_edge_candidates(cur_matrix, cur_naive_S)
    
    if args.solver_type == 'greedy_average':
        ori_greedy = baselines["greedy_average"]
        edge_candidates = fc_env.get_edge_candidates(cur_matrix, cur_average_S)
    elif args.solver_type == 'local':
        ori_greedy = baselines["local"]
        edge_candidates = fc_env.get_edge_candidates(cur_matrix, cur_local_S)
    elif args.solver_type == 'gurobi':
        ori_greedy = baselines["gurobi"]
        edge_candidates = fc_env.get_edge_candidates(cur_matrix, cur_gurobi_S)
    
    defense_bs_result = beam_search(defense_net, fc_env, cur_matrix, limit, edge_candidates, ori_greedy, args.max_timesteps, args.search_size, mp_pool, args, True)
    
    # remove black edges
    row, col, val = cur_matrix.coo()
    new_row = deepcopy(row)
    new_col = deepcopy(col)
    new_val = deepcopy(val)
    for act in bs_result["acts"]:
        for i in range(len(row)):
            if new_row[i] == act[0] and new_col[i] == act[1]:
                break
        new_row = torch.cat((new_row[:i],new_row[i+1:]), dim=0)
        new_col = torch.cat((new_col[:i],new_col[i+1:]), dim=0)
        new_val = torch.cat((new_val[:i],new_val[i+1:]), dim=0)
    
    defense_matrix = SparseTensor(new_row, None, new_col, new_val, new_sizes)
    defense_naive_obj, defense_naive_S = greedy_search_naive(defense_matrix, limit)
    defense_average_obj, defense_average_S = greedy_average_search(defense_matrix, limit)
    defense_local_obj, defense_local_S = local_search(defense_matrix, limit)
    defense_gurobi_obj, defense_gurobi_S = gurobi_search(defense_matrix, limit)
    
    print(f'\t \t'
        f'defense_naive {defense_naive_obj:.4f} \t'
        f'defense_average {defense_average_obj:.4f} \t'
        f'defense_local {defense_local_obj:.4f} \t'
        f'defense_gurobi {defense_gurobi_obj:.4f} \t')
    
    return defense_bs_result

def evaluate(policy_net, fc_env, eval_graphs, max_steps=10, search_size=10, mp_pool=None, args = None, defense_net = None):
    ret_result = {'reward': {}, 'ratio': {}, 'solution': {}, 'gap': {}, 'num_act': {}, 'time': {},
                  'each_step_reward': {}, 'each_step_solution': {}, 'each_step_ratio': {}}
    defense_ret_result = {'ratio': {}, 'solution': {},}
    # Load test graphs
    for graph_index, (inp_matrix, limit, edge_candidates, ori_greedy, baselines, _) in enumerate(eval_graphs):
        # Running beam search:
        bs_result = beam_search(policy_net, fc_env, inp_matrix, limit, edge_candidates, ori_greedy, max_steps,
                                search_size, mp_pool, args)
        print(f'BEAMSEARCH \t'
              f'gid {graph_index} \t'
              f'time {bs_result["time"]:.2f} \t'
              f'reward {bs_result["reward"]:.4f} \t'
              f'ratio {bs_result["reward"] / (ori_greedy+1e-4):.4f} \t'
              f'ours {bs_result["solution"]:.4f} \t'
              f'gap {bs_result["solution"] - min([v for v in baselines.values()]):.4f} \t'
              + '\t'.join([f'{key} {val:.4f}' for key,val in baselines.items()]) + '\t'
              f'action {bs_result["acts"]} \t')
        
        # record statistics
        ret_result['reward'][f'graph{graph_index}'] = bs_result['reward']
        ret_result['ratio'][f'graph{graph_index}'] = bs_result["reward"] / (ori_greedy+1e-4)
        ret_result['gap'][f'graph{graph_index}'] = \
            bs_result["solution"] - min([v for v in baselines.values()])
        ret_result['solution'][f'graph{graph_index}_ours'] = bs_result["solution"]
        ret_result['num_act'][f'graph{graph_index}'] = len(bs_result["acts"])
        for key, val in baselines.items():
            ret_result['solution'][f'graph{graph_index}_{key}'] = val
        ret_result['time'][f'graph{graph_index}'] = bs_result['time']
        
        if args.is_defense:
            defense_bs_result = metric_calculate(fc_env, inp_matrix, limit, bs_result, baselines, args, defense_net, mp_pool)
            # record statistics
            defense_ret_result['ratio'][f'graph{graph_index}'] = defense_bs_result["reward"] / (ori_greedy+1e-4)
            defense_ret_result['solution'][f'graph{graph_index}_ours'] = defense_bs_result["solution"]

        # record each step statistics
        ret_result['each_step_reward'][f'graph{graph_index}'] = bs_result['best_reward_each_step']
        ret_result['each_step_ratio'][f'graph{graph_index}'] = bs_result['best_reward_each_step'] / ori_greedy
        ret_result['each_step_solution'][f'graph{graph_index}'] = bs_result['best_solution_each_step']
            
    # compute mean
    for key, val in ret_result.items():
        if key == 'solution':
            ours_vals = []
            for sol_key, sol_val in val.items():
                if 'ours' in sol_key:
                    ours_vals.append(sol_val)
            ret_result[key]['mean'] = sum(ours_vals) / len(ours_vals)
        elif 'each' not in key:
            ret_result[key]['mean'] = sum(val.values()) / len(val)
        else:
            print(key, ret_result[key].keys())
            step_mean = np.zeros(len(ret_result[key]['graph0']))
            step_array = dict()
            n = 0
            for key2, val2 in ret_result[key].items():
                n = len(val2)
                for idx in range(len(val2)):
                    step_mean[idx] += val2[idx] / len(ret_result[key])

                    if idx in step_array:
                        step_array[idx].append(val2[idx])
                    else:
                        step_array[idx] = []
            ret_result[key]['mean'] = step_mean
            ret_result[key]['std'] = []
            for idx in range(n):
                ret_result[key]['std'].append(np.std(step_array[idx], ddof=1))

    print(f'BEAMSEARCH \t solution {ret_result["solution"]["mean"]:.4f} \t'
        f' ratio percent {ret_result["ratio"]["mean"]:.4f}')
    
    if args.is_defense:
        for key, val in defense_ret_result.items():
            if key == 'solution':
                defense_ours_vals = []
                for sol_key, sol_val in val.items():
                    if 'ours' in sol_key:
                        ours_vals.append(sol_val)
                defense_ret_result[key]['mean'] = sum(ours_vals) / len(ours_vals)
            else:
                defense_ret_result[key]['mean'] = sum(val.values()) / len(val)
        print(f'Defense BEAMSEARCH \t solution {defense_ret_result["solution"]["mean"]:.4f} \t'
            f' ratio percent {defense_ret_result["ratio"]["mean"]:.4f}')
    
    return ret_result

if __name__ == "__main__":
    import os, random
    from torch.multiprocessing import Pool, cpu_count
    from fc_ppo_pytorch import parse_arguments, ActorCritic
    from fc_env import FCEnv

    args = parse_arguments()
    # initialize manual seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # create environment
    fc_env = FCEnv(args.solver_type, args.strategy_size, args.event_size, args.is_attack, args.modify_nodes)
    args.node_feature_dim = 4

    # get current device (cuda or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load training/testing data
    tuples_train, tuples_test = fc_env.generate_tuples(args.train_sample, args.test_sample, 0)

    # load attack/defense models
    ac_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers, args.modify_nodes

    # load the defense model
    defense_filename = f'PPO_{args.solver_type}_strategy{args.strategy_size}_event{args.event_size}' \
                       f'_beam{args.search_size}_ratio'
    for file in os.listdir(args.pretrained_save_dir):
        if defense_filename in file:
            defense_filename = args.pretrained_save_dir + '/' + file
            break
    print(f'Defense file {defense_filename}')
    defense_policy = ActorCritic(*ac_params).to(device)
    defense_state_dict = torch.load(defense_filename)
    defense_policy.load_state_dict(defense_state_dict)
    for param in defense_policy.parameters():
        param.requires_grad = False

    # load the attack model
    attack_filename = f'PPO_{args.solver_type}attack_strategy{args.strategy_size}_event{args.event_size}' \
                      f'_beam{args.search_size}_ratio'
    for file in os.listdir(args.pretrained_save_dir):
        if attack_filename in file:
            attack_filename = args.pretrained_save_dir + '/' + file
            break
    print(f'Attack file {attack_filename}')
    attack_policy = ActorCritic(*ac_params).to(device)
    attack_state_dict = torch.load(attack_filename)
    attack_policy.load_state_dict(attack_state_dict)
    for param in attack_policy.parameters():
        param.requires_grad = False

    # num_workers = cpu_count()
    # mp_pool = Pool(num_workers)
    mp_pool = None

    print("########## Evaluate on Test ##########")
    ret_result = evaluate(attack_policy, fc_env, tuples_test, args.max_timesteps, args.search_size, mp_pool, args=args,
                         defense_net=defense_policy)
    print("########## Evaluate complete ##########\n", flush=True)

    for key, value in ret_result.items():
        print(key, value)

    record_item = {"each_step_ratio" : ret_result['each_step_ratio']['mean'],
                   "each_step_rewrad": ret_result['each_step_reward']['mean'],
                   "each_step_ratio_std" : ret_result['each_step_ratio']['std'],
                   "each_step_rewrad_std" : ret_result['each_step_reward']['std']}

    record_name = f"FC_{args.solver_type}_{args.strategy_size}"
    import pickle
    file_in = open("result/FC_Steps(K).pkl", "rb")
    record = pickle.load(file_in)
    file_in.close()

    # record = dict()
    file_out = open("result/FC_Steps(K).pkl", "wb")
    record[record_name] = record_item
    pickle.dump(record, file_out)