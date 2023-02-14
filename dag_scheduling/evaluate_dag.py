import sys
sys.path.append("../")

import time
import itertools
import torch
import random
import numpy as np

def repeat_interleave(inp_list, repeat_num):
    return list(itertools.chain.from_iterable(zip(*itertools.repeat(inp_list, repeat_num))))


def beam_search_step_kernel(idx, act_n_sel,
                            acts1, acts2, probs1, probs2, ready_nodes1, ready_nodes2_flat,
                            graph_list, act_list, prob_list, orig_greedy, dag_model, defense = False):
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
        reward, new_graph, new_greedy, edge_candidates, done = \
            dag_model.step(graph_list[beam_idx], (act1, act2), orig_greedy, defense)
        return (
                new_graph,
                reward,
                act_list[beam_idx] + [(act1, act2)],
                prob_list[beam_idx] + [(prob1, prob2)],
                edge_candidates,
                done
        )
    else:
        return None


def beam_search(policy_model, dag_model, inp_graph, greedy_cost, max_actions, beam_size=5, multiprocess_pool=None, args = None, defense = False):
    start_time = time.time()

    state_encoder = policy_model.state_encoder
    actor_net = policy_model.actor_net

    graph_copy = inp_graph.copy()
    orig_greedy = greedy_cost
    best_tuple = (
        graph_copy,  # graph
        0,  # accumulated reward
        [],  # actions
        [],  # probabilities
        dag_model.get_edge_candidates(graph_copy, defense),  # edge candidates
        False,  # stop flag
    )
    topk_graphs = [best_tuple]

    act_n_sel = beam_size
    best_reward_each_step = np.zeros(max_actions + 1)
    for step in range(1, max_actions + 1):
        graph_list, reward_list, act_list, prob_list, edge_cand_list = [], [], [], [], []
        for graph, reward, acts, probs, edge_cand, done in topk_graphs:
            assert done is False
            graph_list.append(graph)
            reward_list.append(reward)
            act_list.append(acts)
            prob_list.append(probs)
            edge_cand_list.append(edge_cand)
        # print(len(graph_list))
        # print(graph_list)
        if len(graph_list) == 0:
            break

        state_feat = state_encoder(graph_list)

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
                    graph_list, act_list, prob_list,
                    orig_greedy, dag_model, defense
                )

        if multiprocess_pool:
            pool_map = multiprocess_pool.starmap_async(
                beam_search_step_kernel, kernel_func_feeder(len(graph_list) * act_n_sel ** 2))
            tmp_graphs = pool_map.get()
        else:
            tmp_graphs = [beam_search_step_kernel(*x) for x in kernel_func_feeder(len(graph_list) * act_n_sel ** 2)]
        searched_graphs = []
        for graph_tuple in tmp_graphs:
            if graph_tuple is not None:
                searched_graphs.append(graph_tuple)

        # find the best action
        searched_graphs.sort(key=lambda x: x[1], reverse=True)
        if searched_graphs[0][1] > best_tuple[1]:
            best_tuple = searched_graphs[0]

        best_reward_each_step[step] = best_tuple[1]
        # find the topk expandable actions
        topk_graphs = []
        for g in searched_graphs[:beam_size]:
            if not g[5]:
                topk_graphs.append(g)

    ret_solution = orig_greedy - best_tuple[1]
    best_solution_each_step = orig_greedy - best_reward_each_step

    if args.is_attack and not defense:
        ret_solution = orig_greedy + best_tuple[1]
        best_solution_each_step = orig_greedy + best_reward_each_step

    return {
        'reward': best_tuple[1],
        'solution': ret_solution,
        'acts': best_tuple[2],
        'probs': best_tuple[3],
        'time': time.time() - start_time,
        'best_reward_each_step': best_reward_each_step,
        'best_solution_each_step': best_solution_each_step
    }

def metric_calculate(dag_graph, inp_graph, bs_result, baselines, args, defense_net = None, mp_pool = None):
    cur_graph = inp_graph.copy()
    for act in bs_result["acts"]:
            cur_graph.remove_edge(act[0], act[1], )

    sfs = dag_graph.shortest_first_scheduling(cur_graph)
    cps = dag_graph.critical_path_scheduling(cur_graph)
    ts = dag_graph.tetris_scheduling(cur_graph)
    
    sfs_ratio = (sfs - baselines["shortest_first"]) / baselines["shortest_first"]
    cp_ratio = (cps - baselines["critical_path"]) / baselines["critical_path"]
    ts_ratio = (ts - baselines["tetris"]) / baselines["tetris"]
    
    print(f'\t \t'
        f'cur_sfs {sfs:.4f} \t'
        f'cur_sfs_ratio {sfs_ratio:.4f} \t'
        f'cur_cp {cps:.4f} \t'
        f'cur_cp_ratio {cp_ratio:.4f} \t'
        f'cur_ts {ts:.4f} \t'
        f'cur_ts_ratio {ts_ratio:.4f} \t')
    
    ori_greedy = baselines["shortest_first"]
    if args.scheduler_type == "cp":
        ori_greedy = baselines["critical_path"]
    elif args.scheduler_type == "ts":
        ori_greedy = baselines["tetris"]
    defense_bs_result = beam_search(defense_net, dag_graph, cur_graph, ori_greedy, args.max_timesteps, args.search_size, mp_pool, args, True)
    
    for act in defense_bs_result["acts"]:
        cur_graph.add_edge(act[1], act[0])
    
    defense_sfs = dag_graph.shortest_first_scheduling(cur_graph)
    defense_cps = dag_graph.critical_path_scheduling(cur_graph)
    defense_ts = dag_graph.tetris_scheduling(cur_graph)
    
    defense_sfs_ratio = (defense_sfs - baselines["shortest_first"]) / baselines["shortest_first"]
    defense_cp_ratio = (defense_cps - baselines["critical_path"]) / baselines["critical_path"]
    defense_ts_ratio = (defense_ts - baselines["tetris"]) / baselines["tetris"]
    
    print(f'\t \t'
        f'defense_sfs {defense_sfs:.4f} \t'
        f'defense_sfs_ratio {defense_sfs_ratio:.4f} \t'
        f'defense_cp {defense_cps:.4f} \t'
        f'defense_cp_ratio {defense_cp_ratio:.4f} \t'
        f'defense_ts {defense_ts:.4f} \t'
        f'defense_ts_ratio {defense_ts_ratio:.4f} \t')

    return defense_bs_result

def evaluate(policy_net, dag_graph, eval_graphs, max_steps=10, search_size=10, mp_pool=None, args = None, defense_net = None):
    ret_result = {'reward': {}, 'ratio': {}, 'solution': {}, 'gap': {}, 'num_act': {}, 'time': {},
                  'each_step_reward':{}, 'each_step_solution':{}, 'each_step_ratio':{}}
    defense_ret_result = {'ratio': {}, 'solution': {},}
    # Load test graphs
    for graph_index, (inp_graph, ori_greedy, _, baselines) in enumerate(eval_graphs):
        # Running beam search:
        bs_result = beam_search(policy_net, dag_graph, inp_graph, ori_greedy, max_steps, search_size, mp_pool, args)
        print(f'BEAMSEARCH \t'
              f'gid {graph_index} \t'
              f'time {bs_result["time"]:.2f} \t'
              f'reward {bs_result["reward"]:.4f} \t'
              f'ratio {bs_result["reward"] / ori_greedy:.4f} \t'
              f'ours {bs_result["solution"]:.4f} \t'
              f'gap {(bs_result["solution"] - min([v for v in baselines.values()])) / bs_result["solution"]:.4f} \t'
              f'sfs {baselines["shortest_first"]:.4f} \t'
              f'cp {baselines["critical_path"]:.4f} \t'
              f'ts {baselines["tetris"]:.4f} \t'
              f'action {bs_result["acts"]} \t'
              f'prob {",".join([f"({x[0]:.3f}, {x[1]:.3f})" for x in bs_result["probs"]])}')

        # record statistics
        ret_result['reward'][f'graph{graph_index}'] = bs_result['reward']
        ret_result['ratio'][f'graph{graph_index}'] = bs_result["reward"] / ori_greedy
        ret_result['gap'][f'graph{graph_index}'] = \
            (bs_result["solution"] - min([v for v in baselines.values()])) / bs_result["solution"]
        ret_result['solution'][f'graph{graph_index}_ours'] = bs_result["solution"]
        ret_result['num_act'][f'graph{graph_index}'] = len(bs_result["acts"])
        for key, val in baselines.items():
            ret_result['solution'][f'graph{graph_index}_{key}'] = val
        ret_result['time'][f'graph{graph_index}'] = bs_result['time']

        # record the each step statistics
        ret_result['each_step_reward'][f'graph{graph_index}'] = bs_result['best_reward_each_step']
        ret_result['each_step_ratio'][f'graph{graph_index}'] = bs_result["best_reward_each_step"] / ori_greedy
        ret_result['each_step_solution'][f'graph{graph_index}'] = bs_result["best_solution_each_step"]
        
        if args.is_defense:
            defense_bs_result = metric_calculate(dag_graph, inp_graph, bs_result, baselines, args, defense_net, mp_pool)
            # record statistics
            defense_ret_result['ratio'][f'graph{graph_index}'] = defense_bs_result["reward"] / (ori_greedy+1e-4)
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
          f' mean ratio {ret_result["ratio"]["mean"]:.4f}', end='\t\t')

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


if __name__ == '__main__':
    import os
    from torch.multiprocessing import Pool, cpu_count

    from dag_graph import DAGraph
    from dag_scheduling.dag_data.dag_generator import load_tpch_tuples
    from dag_scheduler_ppo_pytorch import ActorCritic, parse_arguments

    args = parse_arguments()

    # initialize manual seed
    if args.random_seed != None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # create DAG graph environment
    resource_dim = 1
    raw_node_feature_dim = 1 + resource_dim  # (duration, resources)
    args.node_feature_dim = raw_node_feature_dim
    dag_graph = DAGraph(resource_dim=resource_dim,
                        feature_dim=args.node_feature_dim,
                        scheduler_type=args.scheduler_type,
                        is_attack = args.is_attack)

    # load training/testing data
    vargs = (
        dag_graph,
        args.num_init_dags,
        raw_node_feature_dim,
        resource_dim,
        args.resource_limit,
        args.add_graph_features,
        args.scheduler_type
    )
    tuples_train, tuples_test = \
        load_tpch_tuples(args.train_sample, 0, *vargs), load_tpch_tuples(args.test_sample, 1, *vargs)

    # get current device (cuda or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init models
    ac_params = dag_graph, args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, \
                args.gnn_layers


    # policy_net = ActorCritic(*ac_params).to(device)
    # policy_net.load_state_dict(torch.load(args.test_model_weight))
    # num_workers = cpu_count()
    # mp_pool = Pool(num_workers)

    # load the attack model
    attack_filename = f'PPO_{args.scheduler_type}attack_dag_num{args.num_init_dags}' \
                      f'_beam{args.search_size}'
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
    ret_result = evaluate(attack_policy, dag_graph, tuples_test, args.max_timesteps, args.search_size, mp_pool, args)
    print("########## Evaluate complete ##########\n", flush=True)

    for key, value in ret_result.items():
        print(key, value)

    record_item = {"each_step_ratio" : ret_result['each_step_ratio']['mean'],
                   "each_step_rewrad": ret_result['each_step_reward']['mean'],
                   "each_step_ratio_std" : ret_result['each_step_ratio']['std'],
                   "each_step_rewrad_std" : ret_result['each_step_reward']['std']}

    record_name = f"DAG_{args.scheduler_type}_{args.num_init_dags}"
