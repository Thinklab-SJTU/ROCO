import time
import os
import itertools
import torch
from copy import deepcopy
from tsp_algorithms import calc_furthest_insertion_tour_len, calc_lkh_tour_len, calc_nearest_neighbor_tour_len,\
    get_lower_matrix, solveFarthestInsertion, calc_MatNet_tour_len
import tsplib95
from tsp_main import parse_tsp
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VERY_LARGE_INT = 10 # 65536

def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))


def beam_search_step_kernel(idx, act_n_sel,
                            acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                            graph_list, act_list, prob_list, orig_greedy, tsp_env, defense = False):
    beam_idx = idx // act_n_sel ** 2
    act1_idx = idx // act_n_sel % act_n_sel
    act2_idx = idx % act_n_sel
    act1, prob1 = acts1[beam_idx, act1_idx].item(), probs1[beam_idx, act1_idx].item()
    act2, prob2 = acts2[beam_idx, act1_idx, act2_idx].item(), probs2[beam_idx, act1_idx, act2_idx].item()
    ready_nodes_1 = ready_nodes1[beam_idx]
    ready_nodes_2 = ready_nodes2_flat[beam_idx * act_n_sel + act1_idx]

    if act1 in ready_nodes_1 and act2 in ready_nodes_2:
        assert prob1 > 0
        assert prob2 > 0
        reward, new_lower_matrix, edge_candidates, new_greedy, done = \
            tsp_env.step(graph_list[beam_idx], (act1, act2), orig_greedy, defense)
        return (
                new_lower_matrix,
                edge_candidates,
                reward,
                act_list[beam_idx] + [(act1, act2)],
                prob_list[beam_idx] + [(prob1, prob2)],
                done
        )
    else:
        return None

def beam_search(policy_model, tsp_env, inp_lower_matrix, edge_candidates, greedy_cost, max_actions, beam_size=5, multiprocess_pool=None, args = None, defense = False):
    start_time = time.time()

    state_encoder = policy_model.state_encoder
    actor_net = policy_model.actor_net

    orig_greedy = greedy_cost
    best_tuple = (
        deepcopy(inp_lower_matrix),  # input lower-left adjacency matrix
        edge_candidates,  # edge candidates
        -100,  # accumulated reward
        [],  # actions
        [],  # probabilities
        False,
    )
    topk_graphs = [best_tuple]

    act_n_sel = beam_size
    best_reward_each_step = np.zeros(max_actions + 1)
    for step in range(1, max_actions+1):
        lower_matrix_list, edge_cand_list, reward_list, act_list, prob_list = [], [], [], [], []
        for lower_matrix, edge_cand, reward, acts, probs, done in topk_graphs:
            lower_matrix_list.append(lower_matrix)
            edge_cand_list.append(edge_cand)
            reward_list.append(reward)
            act_list.append(acts)
            prob_list.append(probs)
            if done:
                ret_solution = orig_greedy + reward if (args.is_attack and not defense) else orig_greedy - reward
                return {
                    'reward': reward,
                    'solution': ret_solution,
                    'acts': acts,
                    'probs': probs,
                    'time': time.time() - start_time,
                }

        state_feat = state_encoder(lower_matrix_list)

        # mask1: (beam_size, max_num_nodes)
        mask1, ready_nodes1 = actor_net._get_mask1(state_feat.shape[0], state_feat.shape[1], edge_cand_list)
        # acts1, probs1: (beam_size, act_n_sel)
        acts1, probs1 = actor_net._select_node(state_feat, mask1, greedy_sel_num=act_n_sel)
        # acts1_flat, probs1_flat: (beam_size x act_n_sel,)
        acts1_flat, probs1_flat = acts1.reshape(-1), probs1.reshape(-1)
        # mask2_flat: (beam_size x act_n_sel, max_num_nodes)
        mask2_flat, ready_nodes2_flat = actor_net._get_mask2(
            state_feat.shape[0] * act_n_sel, state_feat.shape[1], repeat_interleave(edge_cand_list, act_n_sel),
            acts1_flat)
        # acts2_flat, probs2_flat: (beam_size x act_n_sel, act_n_sel)
        acts2_flat, probs2_flat = actor_net._select_node(
            state_feat.repeat_interleave(act_n_sel, dim=0), mask2_flat, prev_act=acts1_flat, greedy_sel_num=act_n_sel)
        # acts2, probs2: (beam_size, act_n_sel, act_n_sel)
        acts2, probs2 = acts2_flat.reshape(-1, act_n_sel, act_n_sel), probs2_flat.reshape(-1, act_n_sel, act_n_sel)

        acts1, acts2, probs1, probs2 = acts1.cpu(), acts2.cpu(), probs1.cpu(), probs2.cpu()

        def kernel_func_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                    lower_matrix_list, act_list, prob_list,
                    orig_greedy, tsp_env, defense
                )
        
        # if multiprocess_pool:
        #     pool_map = multiprocess_pool.starmap_async(
        #         beam_search_step_kernel, kernel_func_feeder(len(lower_matrix_list) * act_n_sel ** 2))
        #     tmp_graphs = pool_map.get()
        # else:
        tmp_graphs = [beam_search_step_kernel(*x) for x in kernel_func_feeder(len(lower_matrix_list) * act_n_sel ** 2)]
        searched_graphs = []
        for graph_tuple in tmp_graphs:
            if graph_tuple is not None:
                searched_graphs.append(graph_tuple)

        # find the best action
        searched_graphs.sort(key=lambda x: x[2], reverse=True)
        if searched_graphs[0][2] > best_tuple[2]:
            best_tuple = searched_graphs[0]
        # print(searched_graphs[0], '\n\n')
        best_reward_each_step[step] = best_tuple[2]
        # find the topk expandable actions
        topk_graphs = searched_graphs[:beam_size]

    ret_solution = orig_greedy + best_tuple[2] if (args.is_attack and not defense) else orig_greedy - best_tuple[2]
    best_solution_each_step = orig_greedy + best_reward_each_step if (args.is_attack and not defense) else orig_greedy - best_reward_each_step
    return {
        'inp_lower_matrix': best_tuple[0],
        'reward': best_tuple[2],
        'solution': ret_solution,
        'acts': best_tuple[3],
        'probs': best_tuple[4],
        'time': time.time() - start_time,
        'best_reward_each_step': best_reward_each_step,
        'best_solution_each_step': best_solution_each_step,
    }

def metric_calculate(tsp_env, inp_lower_matrix, bs_result, baselines, args, defense_net = None, mp_pool = None, tester = None):
    cur_lower_matrix = deepcopy(inp_lower_matrix)

    for act in bs_result["acts"]:
        cur_lower_matrix[act[0]][act[1]] /= 2 

    cur_tsp_inst = tsplib95.parse(parse_tsp(cur_lower_matrix))
    cur_nn_tour, cur_nn = calc_nearest_neighbor_tour_len(cur_tsp_inst)
    cur_furthest_tour, cur_furthest = solveFarthestInsertion(cur_tsp_inst)
    cur_lkh_5_tour, cur_lkh_5 = calc_lkh_tour_len(cur_tsp_inst)
    cur_MatNet_tour, cur_MatNet = calc_MatNet_tour_len((torch.tensor(cur_lower_matrix)/1e4).to(torch.float32).to(device), tester)

    print(f'\t \t'
        f'cur_nn {cur_nn:.4f} \t'
        f'cur_furthest {cur_furthest:.4f} \t'
        f'cur_lkh-5 {cur_lkh_5:.4f}\t'
        f'cur_MatNet {cur_MatNet:.4f}\t')

    ori_greedy = baselines['nn']
    num_nodes = len(cur_lower_matrix)
    edge_candidates = tsp_env.edge_candidate_from_tour(cur_nn_tour, num_nodes)
    if args.solver_type == 'furthest':
        ori_greedy = baselines['furthest']
        edge_candidates = tsp_env.edge_candidate_from_tour(cur_furthest_tour, num_nodes)
    elif args.solver_type == 'lkh-5':
        ori_greedy = baselines['lkh-5']
        edge_candidates = tsp_env.edge_candidate_from_tour(cur_lkh_5_tour, num_nodes)
    elif args.solver_type == 'MatNet':
        ori_greedy = baselines['MatNet']
        edge_candidates = tsp_env.edge_candidate_from_tour(cur_MatNet_tour, num_nodes)
    defense_bs_result = beam_search(defense_net, tsp_env, cur_lower_matrix, edge_candidates, ori_greedy, args.max_timesteps, args.search_size, mp_pool, args, True)

    for act in defense_bs_result["acts"]:
        cur_lower_matrix[act[0]][act[1]] *= 2
    
    defense_tsp_inst = tsplib95.parse(parse_tsp(cur_lower_matrix))
    _, defense_nn = calc_nearest_neighbor_tour_len(defense_tsp_inst)
    _, defense_furthest = solveFarthestInsertion(defense_tsp_inst)
    _, defense_lkh_5 = calc_lkh_tour_len(defense_tsp_inst)
    _, defense_MatNet = calc_MatNet_tour_len((torch.tensor(cur_lower_matrix)/1e4).to(torch.float32).to(device), tester)

    print(f'\t \t'
        f'defense_nn {defense_nn:.4f} \t'
        f'defense_furthest {defense_furthest:.4f} \t'
        f'defense_lkh-5 {defense_lkh_5:.4f} \t'
        f'defense_MatNet {defense_MatNet:.4f} \t')
    
    return defense_bs_result

def evaluate(policy_net, tsp_env, eval_graphs, max_steps=10, search_size=3, mp_pool=None, args = None, defense_net = None, tester=None):
    ret_result = {'reward': {}, 'ratio': {}, 'solution': {}, 'gap': {}, 'num_act': {}, 'time': {},
                  'each_step_reward': {}, 'each_step_ratio': {}, 'each_step_solution': {}}
    defense_ret_result = {'ratio': {}, 'solution': {},}
    # Load test graphs
    for graph_index, (inp_lower_matrix, edge_candidates, ori_greedy, baselines, _, tsp_path) in enumerate(eval_graphs):
        # Running beam search:
        bs_result = beam_search(policy_net, tsp_env, inp_lower_matrix, edge_candidates, ori_greedy, max_steps,
                                search_size, mp_pool, args)
        print(f'BEAMSEARCH \t'
              f'gid {graph_index} \t'
              f'time {bs_result["time"]:.2f} \t'
              f'reward {bs_result["reward"]:.4f} \t'
              # f'optimum {1 if bs_result["solution"] == 0 else 0:.4f} \t'
              f'ratio {bs_result["reward"] / (ori_greedy+1e-4):.4f} \t'
              f'ours {bs_result["solution"]:.4f} \t'
              f'gap {bs_result["solution"] - min([v for v in baselines.values()]):.4f} \t'
              + '\t'.join([f'{key} {baselines[key]:.4f}' for key in tsp_env.available_solvers]) + '\t'
              f'action {bs_result["acts"]} \t'
              f'prob [{",".join([f"({x[0]:.3f}, {x[1]:.3f})" for x in bs_result["probs"]])}]')

        if args.is_record_bad_case:
            load_prefix = f"atsp_data/{args.node_dimension}/"
            atsp_filename = tsp_path.split('/')[-1]
            problem = tsplib95.load(load_prefix + atsp_filename)
            matrix = bs_result['inp_lower_matrix']
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    if i == j:
                        matrix[i][j] = 9999999
                    else:
                        matrix[i][j] = int(matrix[i][j] * 100)

            problem.edge_weights = matrix
            save_prefix = f"atsp_data_hard_case/{args.solver_type}/{args.node_dimension}/"
            if not os.path.exists(save_prefix):
                os.makedirs(save_prefix)
            problem.save(save_prefix + atsp_filename)
            print(f"Generate the hard case:  {save_prefix + atsp_filename}")

        # record statistics
        ret_result['reward'][f'graph{graph_index}'] = bs_result['reward']
        # ret_result['optimum'][f'graph{graph_index}'] = 1 if bs_result["solution"] == 0 else 0
        ret_result['ratio'][f'graph{graph_index}'] = bs_result["reward"] / (ori_greedy+1e-4)
        ret_result['gap'][f'graph{graph_index}'] = \
            bs_result["solution"] - min([v for v in baselines.values()])
        ret_result['solution'][f'graph{graph_index}_ours'] = bs_result["solution"]
        ret_result['num_act'][f'graph{graph_index}'] = len(bs_result["acts"])
        for key, val in baselines.items():
            ret_result['solution'][f'graph{graph_index}_{key}'] = val
        ret_result['time'][f'graph{graph_index}'] = bs_result['time']

        # record each step statistics
        ret_result['each_step_reward'][f'graph{graph_index}'] = bs_result['best_reward_each_step']
        ret_result['each_step_ratio'][f'graph{graph_index}'] = bs_result['best_reward_each_step'] / ori_greedy
        ret_result['each_step_solution'][f'graph{graph_index}'] = bs_result['best_solution_each_step']
        
        if args.is_defense:
            defense_bs_result = metric_calculate(tsp_env, inp_lower_matrix, bs_result, baselines, args, defense_net, mp_pool, tester)
            # record statistics
            # defense_ret_result['optimum'][f'graph{graph_index}'] = 1 if defense_bs_result["solution"] == 0 else 0
            defense_ret_result['ratio'][f'graph{graph_index}'] = defense_bs_result['reward'] / (ori_greedy+1e-4)
            defense_ret_result['solution'][f'graph{graph_index}_ours'] = defense_bs_result["solution"]
            
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
        # f' optimum percent {ret_result["optimum"]["mean"]:.4f}')
    
    if args.is_defense:
        for key, val in defense_ret_result.items():
            if key == 'solution':
                defense_ours_vals = []
                for sol_key, sol_val in val.items():
                    if 'ours' in sol_key:
                        defense_ours_vals.append(sol_val)
                defense_ret_result[key]['mean'] = sum(defense_ours_vals) / len(defense_ours_vals)
            else:
                defense_ret_result[key]['mean'] = sum(val.values()) / len(val)
        print(f'Defense BEAMSEARCH \t solution {defense_ret_result["solution"]["mean"]:.4f} \t'
            f' ratio percent {defense_ret_result["ratio"]["mean"]:.4f}')
            # f' optimum percent {defense_ret_result["optimum"]["mean"]:.4f}')
    
    return ret_result

if __name__ == "__main__":
    import os
    import random
    from torch.multiprocessing import Pool, cpu_count

    from tsp_env import TSPEnv
    from tsp_ppo_pytorch import ActorCritic, parse_arguments

    from tsp_main import env_params, model_params, tester_params
    from atsp.ATSPTester import ATSPTester

    args = parse_arguments()
    # initialize manual seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # create environment
    tester = ATSPTester(env_params, model_params, tester_params)
    tsp_env = TSPEnv(args.solver_type, args.node_dimension, is_attack=args.is_attack, tester=tester)
    args.node_feature_dim = 1

    # get current device (cuda or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load training/testing data
    tuples_train, tuples_test = tsp_env.generate_tuples(args.train_sample, args.test_sample, 0, True)

    # load attack models
    ac_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers
    attack_filename = f'PPO_{args.solver_type}attack_node{args.node_dimension}' \
                      f'_beam_{args.search_size}_ratio'
    for file in os.listdir(args.pretrained_save_dir):
        if attack_filename in file:
            attack_filename = args.pretrained_save_dir + '/' + file
            print(f'The attack file name: {attack_filename} \t')
            break
    attack_policy = ActorCritic(*ac_params).to(device)
    attack_state_dict = torch.load(attack_filename)
    attack_policy.load_state_dict(attack_state_dict)
    for param in attack_policy.parameters():
        param.requires_grad = False

    num_workers = cpu_count()
    mp_pool = Pool(num_workers)

    print("########## Evaluate on Test ##########")
    ret_result = evaluate(attack_policy, tsp_env, tuples_test, args.max_timesteps, args.search_size, mp_pool,
                         args=args, defense_net=None, tester=tester)
    print("########## Evaluate complete ##########\n", flush=True)

    for key, value in ret_result.items():
        print(key, value)

    record_item = {"each_step_ratio": ret_result['each_step_ratio']['mean'],
                   "each_step_rewrad": ret_result['each_step_reward']['mean'],
                   "each_step_ratio_std": ret_result['each_step_ratio']['std'],
                   "each_step_rewrad_std": ret_result['each_step_reward']['std']}

    record_name = f"ATSP_{args.solver_type}_{args.node_dimension}"
