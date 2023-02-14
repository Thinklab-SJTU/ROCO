import time
import itertools
import os
import torch
from copy import deepcopy
import numpy as np
from torch_sparse import SparseTensor
from mc_algorithms import greedy_search_naive, ortools_search

def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))

def beam_search_step_kernel(idx, act_n_sel,
                            acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                            graph_list, limit, act_list, prob_list, orig_greedy, mc_env):
    beam_idx = idx // act_n_sel ** 2
    act1_idx = idx // act_n_sel % act_n_sel
    act2_idx = idx % act_n_sel
    act1, prob1 = acts1[beam_idx, act1_idx], probs1[beam_idx, act1_idx].item()
    act2, prob2 = acts2[beam_idx, act1_idx, act2_idx], probs2[beam_idx, act1_idx, act2_idx].item()
    ready_nodes1 = ready_nodes1[beam_idx]
    ready_nodes2 = ready_nodes2_flat[beam_idx * act_n_sel + act1_idx]

    if act1 in ready_nodes1 and act2 in ready_nodes2:
        assert prob1 > 0
        assert prob2 > 0
        reward, new_matrix, edge_candidates, new_greedy, done = \
            mc_env.step(graph_list[beam_idx], limit, (act1, act2), orig_greedy)
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

def beam_search(policy_model, mc_env, inp_matrix, limit, edge_candidates, greedy_cost, max_actions, beam_size=5, multiprocess_pool=None, args=None):
    start_time = time.time()

    state_encoder = policy_model.state_encoder
    actor_net = policy_model.actor_net

    orig_greedy = greedy_cost
    best_tuple = (
        deepcopy(inp_matrix), # input sparse matrix
        edge_candidates, # edge candidates
        0, # accumulated reward
        [], # actions
        [], # probabilities
        False,
    )
    topk_graphs = [best_tuple]

    act_n_sel = beam_size
    best_reward_each_step = np.zeros(max_actions + 1)
    for step in range(1, max_actions+1):
        matrix_list, edge_cand_list, reward_list, act_list, prob_list = [], [], [], [], []
        for matrix, edge_cand, reward, acts, probs, done in topk_graphs:
            if done:
                continue
            matrix_list.append(matrix)
            edge_cand_list.append(edge_cand)
            reward_list.append(reward)
            act_list.append(acts)
            prob_list.append(probs)

        element_state_feat, subset_state_feat = state_encoder(matrix_list)

        mask1, ready_nodes1 = actor_net._get_mask1(subset_state_feat.shape[0], subset_state_feat.shape[1], edge_cand_list)
        acts1, probs1 = actor_net._select_node(subset_state_feat, element_state_feat, mask1, greedy_sel_num=act_n_sel)
        acts1_flat, probs1_flat = acts1.reshape(-1), probs1.reshape(-1)
        mask2_flat, ready_nodes2_flat = actor_net._get_mask2(
            element_state_feat.shape[0] * act_n_sel, element_state_feat.shape[1], repeat_interleave(edge_cand_list, act_n_sel),
            acts1_flat)
        acts2_flat, probs2_flat = actor_net._select_node(
            subset_state_feat.repeat_interleave(act_n_sel, dim=0), element_state_feat.repeat_interleave(act_n_sel, dim=0), mask2_flat, prev_act=acts1_flat, greedy_sel_num=act_n_sel)
        acts2, probs2 = acts2_flat.reshape(-1, act_n_sel, act_n_sel), probs2_flat.reshape(-1, act_n_sel, act_n_sel)
        acts1, acts2, probs1, probs2 = acts1.cpu(), acts2.cpu(), probs1.cpu(), probs2.cpu()

        def kernel_func_feeder(max_idx):
            for idx in range(max_idx):
                yield (
                    idx, act_n_sel,
                    acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                    matrix_list, limit, act_list, prob_list,
                    orig_greedy, mc_env
                )

        # if multiprocess_pool:
        #     pool_map = multiprocess_pool.starmap_async(
        #         beam_search_step_kernel, kernel_func_feeder(len(matrix_list) * act_n_sel ** 2))
        #     tmp_graphs = pool_map.get()
        # else:
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

    ret_solution = orig_greedy - best_tuple[2]
    best_solution_each_step = orig_greedy - best_reward_each_step

    return {
        'inp_matrix': best_tuple[0],
        'reward': best_tuple[2],
        'solution': ret_solution,
        'acts': best_tuple[3],
        'probs': best_tuple[4],
        'time': time.time() - start_time,
        'best_reward_each_step': best_reward_each_step,
        'best_solution_each_step': best_solution_each_step
    }

def evaluate(policy_net, mc_env, eval_graphs, max_steps=10, search_size=10, mp_pool=None, args=None):
    ret_result = {'reward': {}, 'ratio': {}, 'solution': {}, 'gap': {}, 'num_act': {}, 'time': {},
                  'each_step_reward':{}, 'each_step_solution':{}, 'each_step_ratio':{}}
    # Load test graphs
    for graph_index, (inp_matrix, limit, edge_candidates, ori_greedy, baselines, _) in enumerate(eval_graphs):
        # Running beam search
        bs_result = beam_search(policy_net, mc_env, inp_matrix, limit, edge_candidates, ori_greedy, max_steps,
                                search_size, mp_pool, args)
        print(f'BEAMSEARCH \t'
              f'gid {graph_index} \t'
              f'time {bs_result["time"]:.2f} \t'
              f'reward {bs_result["reward"]:.4f} \t'
              f'ratio {bs_result["reward"] / (ori_greedy + 1e-4):.4f} \t'
              f'ours {bs_result["solution"]:.4f} \t'
              f'gap {bs_result["solution"] - min([v for v in baselines.values()]):.4f} \t'
              + '\t'.join([f'{key} {val:.4f}' for key, val in baselines.items()]) + '\t'
              f'action {bs_result["acts"]} \t'
              f'prob {",".join([f"({x[0]:.3f}, {x[1]:.3f})" for x in bs_result["probs"]])}')

        if args.is_record_hard_case:
            inp_matrix = bs_result['inp_matrix']
            row, col, val = inp_matrix.coo()

            save_prefix = f'mc_data_hard_case/{args.solver_type}/mc_{args.subset_size}_{args.element_size}'
            mc_filename = f'mc_{args.subset_size}_{args.element_size}_{graph_index+51}.txt'
            if not os.path.exists(save_prefix):
                os.makedirs(save_prefix)
            print(f"Generate the hard case: {os.path.join(save_prefix, mc_filename)}")
            with open(os.path.join(save_prefix, mc_filename), 'w') as f:
                f.write(str(args.subset_size) + ', ' + str(args.element_size) + ', ' +str(limit) + '\n' )
                for i in range(row.shape[0]):
                    ret = [graph_index+1, col[i].item(), row[i].item(), round(val[i].item(),5)]
                    st = ''
                    for ele in ret:
                        st = st + str(ele) + ", "
                    st = st[:-2] + '\n'
                    f.write(st)

        # record statistics
        ret_result['reward'][f'graph{graph_index}'] = bs_result['reward']
        ret_result['ratio'][f'graph{graph_index}'] = bs_result["reward"] / (ori_greedy + 1e-4)
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

    print(f'BEAMSEARCH \t solution {ret_result["solution"]["mean"]:.1f} \t'
          f' ratio percent {ret_result["ratio"]["mean"]:.4f}')

    return ret_result

if __name__ == '__main__':
    import os, random
    from torch.multiprocessing import Pool, cpu_count
    from mc_ppo_pytorch import ActorCritic, parse_arguments
    from mc_env import MCEnv

    args = parse_arguments()
    # initialize manual seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # create environment
    mc_env = MCEnv(args.solver_type, args.subset_size, args.element_size, args.time_limit)
    args.node_feature_dim = 3

    # get current device (cuda or cpu)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # load training/testing data
    tuples_train, tuples_test = mc_env.generate_tuples(args.train_sample, args.test_sample, 0)

    # load attack models
    ac_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers
    attack_filename = f'PPO_{args.solver_type}_subset{args.subset_size}_element{args.element_size}' \
                      f'_beam{args.search_size}_ratio'
    for file in os.listdir(args.pretrained_save_dir):
        if attack_filename in file:
            attack_filename = args.pretrained_save_dir + '/' + file
            break
    print(f"Attack file {attack_filename}")
    attack_policy = ActorCritic(*ac_params).to(device)
    attack_state_dict = torch.load(attack_filename)
    attack_policy.load_state_dict(attack_state_dict)
    for param in attack_policy.parameters():
        param.retains_grad = False

    num_workers = cpu_count()
    mp_pool = Pool(num_workers)
    print("########## Evaluate on Test ##########")
    ret_result = evaluate(attack_policy, mc_env, tuples_test, args.max_timesteps, args.search_size, mp_pool, args=args)
    print("########## Evaluate complete ##########\n", flush=True)

    for key, value in ret_result.items():
        print(key, value)

    record_item = {"each_step_ratio" : ret_result['each_step_ratio']['mean'],
                   "each_step_rewrad": ret_result['each_step_reward']['mean'],
                   "each_step_ratio_std" : ret_result['each_step_ratio']['std'],
                   "each_step_rewrad_std" : ret_result['each_step_reward']['std']}

    record_name = f"MC_{args.solver_type}_{args.subset_size}"