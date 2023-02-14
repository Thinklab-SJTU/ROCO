import sys
sys.path.append("../")

import torch
import numpy as np

from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model

class ATSPTester:
    def __init__(self, env_params, model_params, tester_params):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        
        # cuda
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
        self.device = device
        
        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        
        # Restore
        model_load = self.tester_params['model_load']
        node_cnt = self.env_params['node_cnt']; epoch = model_load['epoch']; path = model_load['path']
        checkpoint_fullname = f'{path}/atsp/checkpoint-{node_cnt}-{epoch}.pt'
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def run(self, problem):
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            problem = problem.repeat(aug_factor, 1, 1)
        else:
            aug_factor = 1
            problem.unsqueeze(0)
        
        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(problem)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
        
            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)
            
            # shape(batch, pomo, 0~)
            tour = self.env.selected_node_list

            # Return
            ###############################################
            aug_reward = reward.reshape(aug_factor, 1, self.env.pomo_size)
            tour = tour.reshape(aug_factor, 1, self.env.pomo_size, -1).cpu().numpy()
            max_pomo_reward, max_pomo_indices = aug_reward.max(dim=2)
            # max_pomo_tour = tour[:, 0, max_pomo_reward, :]
            # no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
            max_aug_pomo_reward, max_aug_pomo_indices = max_pomo_reward.max(dim=0)  # get best results from augmentation
            aug_score = -max_aug_pomo_reward[0].float()  # negative sign to make positive value
            max_pomo_indices = max_pomo_indices.squeeze().cpu().numpy()
            max_aug_pomo_indices = max_aug_pomo_indices.squeeze().cpu().numpy()
            tour = [tour[i,0,max_pomo_indices[i],:] for i in range(len(max_pomo_indices))]
            tour = list(tour[max_aug_pomo_indices])
            tour.append(tour[0])
            return tour, aug_score.cpu().item()