import time
from collections import deque
import gym
import os

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from envs import create_atari_env
from model import ActorCritic
import numpy as np
import pandas as pd
from collections import defaultdict

def episode_trigger(episode_id):
    #return True  # Record every episode
    return episode_id % 100 == 0 # Record every 100 episode
    #return episode_id in [5, 15] # Record at 5 and 15 episode


def test(rank, args, shared_model, counter):
    #torch.manual_seed(args.seed + rank)

    #logging for tensorboard
    #writer = SummaryWriter(log_dir=f"logs/test_worker_{rank}")
    env = create_atari_env(args.env_name)
    #env.seed(args.seed + rank)
    env = gym.wrappers.RecordVideo(env, video_folder='videos', episode_trigger=episode_trigger)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    path='results/'
    if not os.path.exists(path):
        os.makedirs(path)

    state, _ = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    v_all, pi_all = [],[]
    res = defaultdict(list)


    # a quick hack to prevent the agent from stucking
    #actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1

        #debugging, collect datas and save results
        if episode_length%1000==0:
            #np.save(path+'test_v_all.npy',v_all)
            #np.save(path+'test_pi_all.npy',pi_all)
            res_pd = pd.DataFrame.from_dict(res)
            res_pd.to_csv(path+'_result.csv',index=False)
            #print('test v,pi saved!! results saved!')

        # Log shared model parameters
        '''
        if episode_length % 100 == 0:
            for name, param in shared_model.named_parameters():
                writer.add_histogram(f'SharedModelParams/{name}', param.clone().cpu().data.numpy(), episode_length)
                print('shared_model parameter logged!')
        '''
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

        prob = F.softmax(logit, dim=-1)
        v_all.append(value.detach().numpy()[0,0])
        pi_all.append(prob.detach().numpy()[0])

        #action = prob.max(1, keepdim=True)[1].numpy()
        action = prob.multinomial(num_samples=1).detach()
        action_scalar=action.item()

        #five outputs of env.step()
        state, reward, done, _, info = env.step(action_scalar)
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        #actions.append(action[0, 0])
        #if actions.count(actions[0]) == actions.maxlen:
        #    done = True

        if done:

            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            res['r'].append(reward_sum)
            res['stp'].append(episode_length)
            reward_sum = 0
            episode_length = 0
            #actions.clear()
            state, _ = env.reset()
            #time.sleep(60)


        state = torch.from_numpy(state)
