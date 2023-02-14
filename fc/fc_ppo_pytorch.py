import argparse
import torch
from torch import nn
import os
import time
import yaml
import random
import numpy as np
from torch.multiprocessing import Pool, cpu_count
from copy import deepcopy

from fc_model import ActorNet, CriticNet, GraphEncoder
from utils.utils import print_args
from utils.tfboard_helper import get_pai_tensorboard_path, TensorboardUtil
from fc.evaluate_fc import evaluate
from fc_env import FCEnv

class ItemsContainer:
    def __init__(self):
        self.__reward = []
        self.__matrix = []
        self.__limit = []
        self.__ori_greedy = []
        self.__greedy = []
        self.__edge_candidates = []
        self.__done = []

    def append(self, reward, matrix, limit, edge_candidates, greedy, done, ori_greedy):
        self.__reward.append(reward)
        self.__matrix.append(matrix)
        self.__limit.append(limit)
        self.__greedy.append(greedy)
        self.__edge_candidates.append(edge_candidates)
        self.__done.append(done)
        self.__ori_greedy.append(ori_greedy)

    @property
    def reward(self):
        return deepcopy(self.__reward)

    @property
    def matrix(self):
        return deepcopy(self.__matrix)
    
    @property
    def limit(self):
        return deepcopy(self.__limit)

    @property
    def greedy(self):
        return deepcopy(self.__greedy)

    @property
    def edge_candidates(self):
        return deepcopy(self.__edge_candidates)

    @property
    def done(self):
        return deepcopy(self.__done)

    @property
    def ori_greedy(self):
        return deepcopy(self.__ori_greedy)

    def update(self, idx, reward=None, matrix=None, limit=None, greedy=None, edge_candidates=None, done=None, ori_greedy=None):
        if reward is not None:
            self.__reward[idx] = reward
        if matrix is not None:
            self.__matrix[idx] = matrix
        if limit is not None:
            self.__limit[idx] = limit
        if greedy is not None:
            self.__greedy[idx] = greedy
        if edge_candidates is not None:
            self.__edge_candidates[idx] = edge_candidates
        if done is not None:
            self.__done[idx] = done
        if ori_greedy is not None:
            self.__ori_greedy[idx] = ori_greedy
            
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.edge_candidates = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.edge_candidates[:]
        
class ActorCritic(nn.Module):
    def __init__(self, node_feature_dim, node_output_size, batch_norm, one_hot_degree, gnn_layers, modify_nodes=False):
        super(ActorCritic, self).__init__()
        self.state_encoder = GraphEncoder(node_feature_dim, node_output_size, batch_norm, one_hot_degree, gnn_layers)
        self.actor_net = ActorNet(node_output_size * 2, batch_norm, modify_nodes)
        self.value_net = CriticNet(node_output_size * 4, batch_norm)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, matrix, edge_candidates, memory):
        event_feat, strategy_feat = self.state_encoder(matrix)
        actions, action_logits, entropy = self.actor_net(event_feat, strategy_feat, edge_candidates)
        
        memory.states.append(matrix)
        memory.edge_candidates.append(edge_candidates)
        memory.actions.append(actions)
        memory.logprobs.append(action_logits)
        
        return actions

    def evaluate(self, matrix, edge_candidates, action):
        event_feat, strategy_feat = self.state_encoder(matrix)
        _, action_logits, entropy = self.actor_net(event_feat, strategy_feat, edge_candidates, action)
        state_value = self.value_net(event_feat, strategy_feat)
        
        return action_logits, state_value, entropy

class PPO:
    def __init__(self, args, device):
        self.lr = args.learning_rate
        self.betas = args.betas
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.K_epochs = args.k_epochs
        
        self.device = device
        
        ac_params = args.node_feature_dim, args.node_output_size, args.batch_norm, args.one_hot_degree, args.gnn_layers, args.modify_nodes
        
        self.policy = ActorCritic(*ac_params).to(self.device)
        self.optimizer = torch.optim.Adam(
            [{'params': self.policy.actor_net.parameters()},
             {'params': self.policy.value_net.parameters()},
             {'params': self.policy.state_encoder.parameters(), 'lr': self.lr / 10}],
            lr=self.lr, betas=self.betas)
        if len(args.lr_steps) > 0:
            lr_steps = [step // args.update_timestep for step in args.lr_steps]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, lr_steps, gamma=0.1)
        else:
            self.lr_scheduler = None
        self.policy_old = ActorCritic(*ac_params).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def update(self, memory):
        # Time Difference estimate of state rewards:
        rewards = []
        
        with torch.no_grad():
            logprobs, state_values, dist_entropy = \
                self.policy.evaluate(memory.states[-1], memory.edge_candidates[-1], memory.actions[-1].to(self.device))
        discounted_reward = state_values
        
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            discounted_reward = discounted_reward * (1 - torch.tensor(is_terminal, dtype=torch.float32).to(self.device))
            discounted_reward = reward + (self.gamma * discounted_reward).clone()
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards:
        rewards = torch.cat(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = []
        for state in memory.states:
            old_states += state
        old_actions = torch.cat(memory.actions, dim=-1)
        old_logprobs = torch.cat(memory.logprobs, dim=-1)
        old_candidates = []
        for candi in memory.edge_candidates:
            old_candidates += candi
        
        critic_loss_sum = 0
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_candidates, old_actions)

            # Finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Normalizing advantages
            advantages = rewards - state_values.detach()
            #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            # print(state_values, rewards,flush=True)
            critic_loss = self.MseLoss(state_values, rewards)
            entropy_reg = -0.01 * dist_entropy
            critic_loss_sum += critic_loss.detach().mean()

            # take gradient step
            self.optimizer.zero_grad()
            (actor_loss + critic_loss + entropy_reg).mean().backward()
            self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return critic_loss_sum / self.K_epochs  # mean critic loss
            
            
def main(args):
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
    
    # create tensorboard summary writer
    env_dict = os.environ
    print('ENV DICT')
    print(env_dict)
    try:
        import tensorflow as tf
        key = 'POD_NAMESPACE'
        if key in env_dict and env_dict[key] == 'kubemaker':  # kubemaker mode
            tfboard_path = get_pai_tensorboard_path()
        else:  # local mode: logs stored in ./runs/TIME_STAMP-MACHINE_ID
            tfboard_path = args.tensorboard_save_dir
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        tfboard_path = os.path.join(
            tfboard_path, current_time + '_' + socket.gethostname())
        summary_writer = TensorboardUtil(tf.summary.FileWriter(tfboard_path))
    except (ModuleNotFoundError, ImportError):
        print('Warning: Tensorboard not loading, please install tensorflow to enable...')
        summary_writer = None
        
    # init models
    memory = Memory()
    ppo = PPO(args, device)
    num_workers = cpu_count()
    mp_pool = Pool(num_workers)
    
    # logging variables 
    best_test_ratio = 0
    running_reward = 0
    critic_loss = []
    avg_length = 0
    timestep = 0
    prev_time = time.time()
    
    print("Begin Training: ", flush=True)
    # training loop
    for i_episode in range(1, args.max_episodes + 1):
        items_batch = ItemsContainer()
        if i_episode % 10 == 0:
            print(i_episode, flush=True)
        for b in range(args.batch_size):
            graph_index = ((i_episode - 1) * args.batch_size + b) % len(tuples_train)
            matrix, limit, edge_candidates, ori_greedy, baselines, _ = tuples_train[graph_index]
            greedy = ori_greedy
            items_batch.append(0, matrix, limit, edge_candidates, greedy, False, ori_greedy)
        
        for t in range(args.max_timesteps):
            timestep += 1
            
            # Running policy_old:
            with torch.no_grad():
                action_batch = ppo.policy_old.act(items_batch.matrix, items_batch.edge_candidates, memory)
            
            def step_func_feeder(batch_size):
                batch_matrix = items_batch.matrix
                batch_limit = items_batch.limit
                action_batch_cpu = action_batch.cpu()
                batch_greedy = items_batch.greedy
                for b in range(batch_size):
                    yield batch_matrix[b], batch_limit[b], action_batch_cpu[:, b], batch_greedy[b]

            def node_step_func_feeder(batch_size):
                batch_matrix = items_batch.matrix
                batch_limit = items_batch.limit
                action_batch_cpu = action_batch.cpu()
                batch_greedy = items_batch.greedy
                for b in range(batch_size):
                    yield batch_matrix[b], batch_limit[b], action_batch_cpu[b], batch_greedy[b]
                    
            if args.modify_nodes:
                if args.batch_size > 1:
                    pool_map = mp_pool.starmap_async(fc_env.step, node_step_func_feeder(args.batch_size))
                    step_list = pool_map.get()
                else:
                    step_list = [fc_env.step(*x) for x in node_step_func_feeder(args.batch_size)]
            else:
                if args.batch_size > 1:
                    pool_map = mp_pool.starmap_async(fc_env.step, step_func_feeder(args.batch_size))
                    step_list = pool_map.get()
                else:
                    step_list = [fc_env.step(*x) for x in step_func_feeder(args.batch_size)]

            for b, item in enumerate(step_list):
                reward, matrix, edge_candidates, greedy, done = item
                if t == args.max_timesteps - 1:
                    done = True
                items_batch.update(b, reward=reward, matrix=matrix, greedy=greedy,
                                   edge_candidates=edge_candidates, done=done)
            
            # Saving reward and is_terminal:
            memory.rewards.append(items_batch.reward)
            memory.is_terminals.append(items_batch.done)
            
            # update if its time
            if timestep % args.update_timestep == 0:
                closs = ppo.update(memory)
                critic_loss.append(closs)
                if summary_writer:
                    summary_writer.add_scalar('critic mse/train', closs, timestep)
                memory.clear_memory()
            
            running_reward += sum(items_batch.reward) / args.batch_size
            if any(items_batch.done):
                break
            
        avg_length += t+1
        
        # logging
        if i_episode % args.log_interval == 0:
            avg_length = avg_length / args.log_interval
            running_reward = running_reward / args.log_interval
            if len(critic_loss) > 0:
                critic_loss = torch.mean(torch.stack(critic_loss))
            else:
                critic_loss = -1
            now_time = time.time()
            avg_time = (now_time - prev_time) / args.log_interval
            prev_time = now_time
            
            if summary_writer:
                summary_writer.add_scalar('reward/train', running_reward, timestep)
                summary_writer.add_scalar('time/train', avg_time, timestep)
                for lr_id, x in enumerate(ppo.optimizer.param_groups):
                    summary_writer.add_scalar(f'lr/{lr_id}', x['lr'], timestep)
                    
            print(
                f'Episode {i_episode} \t '
                f'avg length: {avg_length:.2f} \t '
                f'critic mse: {critic_loss:.4f} \t '
                f'reward: {running_reward:.4f} \t '
                f'time per episode: {avg_time:.2f}'
            )
            
            running_reward = 0
            avg_length = 0
            critic_loss = []
        
        # testing
        if i_episode % args.test_interval == 0:
            with torch.no_grad():
                prev_test_time = time.time()
                print("######### Evaluate on Test ##########")
                test_dict = evaluate(ppo.policy, fc_env, tuples_test, args.max_timesteps, args.search_size, mp_pool, args)
                
                for key, val in test_dict.items():
                    if isinstance(val, dict):
                        if summary_writer:
                            summary_writer.add_scalars(f'{key}/test', val, timestep)
                    else:
                        if summary_writer:
                            summary_writer.add_scalar(f'{key}/test', val, timestep)
                print("########## Evaluate complete ##########")
                # fix running time value
                prev_time += time.time() - prev_test_time
            
            if test_dict["ratio"]["mean"] > best_test_ratio:
                best_test_ratio = test_dict["ratio"]["mean"]

                file_name = f'{args.checkpoint_save_dir}/PPO_{args.solver_type}_strategy{args.strategy_size}_event{args.event_size}' \
                            f'_beam{args.search_size}_ratio{best_test_ratio:.4f}.pt'
                torch.save(ppo.policy.state_dict(), file_name)
                print(f"Updating the model, Best test_ratio = {best_test_ratio} \n")

def attack_and_defense(args):
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
        
    num_workers = cpu_count()
    mp_pool = Pool(num_workers)
    print("########## Evaluate on Test ##########")
    test_dict = evaluate(attack_policy, fc_env, tuples_test, args.max_timesteps, args.search_size, mp_pool, args=args, defense_net = defense_policy)
    print("########## Evaluate complete ##########\n", flush=True)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='FC solver. You have two ways of setting the parameters: \n'
                                                '1) set parameters by command line arguments \n'
                                                '2) specify --config path/to/config.yaml')
    # environment configs
    parser.add_argument('--solver_type', default='gurobi', choices=['local', 'greedy_naive', 'greedy_average', 'gurobi'])
    parser.add_argument('--resource_limit', default=600, type=float)
    parser.add_argument('--add_graph_features', action='store_true')

    parser.add_argument('--strategy_size', default=30, type=int)
    parser.add_argument('--event_size', default=3000, type=int)
    parser.add_argument('--gamma', default=0.95, type=float, help='discount factor for accumulated reward')
    parser.add_argument('--train_sample', default=50, type=int, help='number of training samples')
    parser.add_argument('--test_sample', default=5, type=int, help='number of testing samples')

    # decode(testing) configs
    parser.add_argument('--search_size', default=6, type=int)

    # learning configs
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--lr_steps', default=[], type=list)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size when sampling')
    parser.add_argument('--betas', default=(0.9, 0.999), help='Adam optimizer\'s beta')
    parser.add_argument('--max_episodes', default=500, type=int, help='max training episodes')
    parser.add_argument('--max_timesteps', default=10, type=int, help='max timesteps in one episode')
    parser.add_argument('--update_timestep', default=20, type=int, help='update policy every n timesteps')
    parser.add_argument('--k_epochs', default=10, type=int, help='update policy for K epochs')
    parser.add_argument('--eps_clip', default=0.1, type=float, help='clip parameter for PPO')

    # model parameters
    parser.add_argument('--one_hot_degree', default=0, type=int) ###################one_hot——degree
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--node_output_size', default=16, type=int)
    parser.add_argument('--gnn_layers', default=3, type=int, help='number of GNN layers')

    # misc configs
    parser.add_argument('--config', default=None, type=str, help='path to config file,'
                        ' and command line arguments will be overwritten by the config file')
    parser.add_argument('--random_seed', default=None, type=int)
    parser.add_argument('--test_interval', default=50, type=int, help='run testing in the interval (episodes)')
    parser.add_argument('--log_interval', default=50, type=int, help='print avg reward in the interval (episodes)')
    parser.add_argument('--test_model_weight', default='', type=str, help='the path of model weight to be loaded')

    # direction
    parser.add_argument('--checkpoint_save_dir', default='checkpoint/0830', type=str, help='the direction to save the checkpoint')
    parser.add_argument('--tensorboard_save_dir', default='logs/tensorboard_log/0830', type=str,
                        help='the direction to save the tensorboard')
    
    # Attack setting
    parser.add_argument('--is_attack', default=True, type=bool, help='Whether to use the attack setting')
    parser.add_argument('--modify_nodes', default=False, type=bool, help="To modify the nodes or edges")

    # Defense setting
    parser.add_argument("--pretrained_save_dir", default="../pretrained/fc", type=str, help='the direction to load the pretrained defense models')
    parser.add_argument("--is_defense", default=False, type=bool, help="run in attack and defense mode")
    
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg_dict = yaml.load(f)
            for key, val in cfg_dict.items():
                assert hasattr(args, key), 'Unknown config key: {key}'
                setattr(args, key, val)
            f.seek(0)
            print('Config file: {args.config}', )
            for line in f.readlines():
                print(line.rstrip())

    # Modify args path
    args.checkpoint_save_dir = args.checkpoint_save_dir + '_' + str(args.strategy_size) + '_' + str(args.event_size) + '_' + args.solver_type
    args.tensorboard_save_dir = args.tensorboard_save_dir + '_' + str(args.strategy_size) + '_' + str(args.event_size) + '_' + args.solver_type
    if args.is_attack:
        args.checkpoint_save_dir = args.checkpoint_save_dir + '_attack'
        args.tensorboard_save_dir = args.tensorboard_save_dir + '_attack'
    if args.modify_nodes:
        args.checkpoint_save_dir = args.checkpoint_save_dir + '_nodes'
        args.tensorboard_save_dir = args.tensorboard_save_dir + '_nodes'
    
    if not os.path.exists(args.checkpoint_save_dir):
        os.makedirs(args.checkpoint_save_dir)
    print_args(args)

    return args


if __name__ == '__main__':
    args = parse_arguments()
    if args.is_defense:
        attack_and_defense(args)
    else:
        main(args)