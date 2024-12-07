import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from envs import create_atari_env
from model import ActorCritic
import math
import numpy as np
import threading
import os



def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    #torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    #env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        print('optimizer is none!')
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    #path='results/'
    #if not os.path.exists(path):
    #    os.makedirs(path)
    #writer = SummaryWriter(log_dir=f"logs/train_worker_{rank}")

    state,_ = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0

    #debugging counters
    allStpCnt = 0
    outStpCnt = 0
    v_all,pi_all = [],[]

    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):

            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1)

            #collect v and pi
            v_all.append(value.detach().numpy()[0,0])
            pi_all.append(prob.detach().numpy()[0])

            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_scalar = action.item()

            state, reward, done, _, info = env.step(action_scalar)

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state, _ = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            #debugging, save learned data
            #if rank == 0 and allStpCnt%10000 == 0:
            #    np.save(path+'train_v_all.npy',v_all)
            #    np.save(path+'train_pi_all.npy',pi_all)
                #print('train v and pi saved!!!')

            allStpCnt += 1

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]


        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Log losses and gradient norms
        '''
        if outStpCnt % 1000 == 0:
            writer.add_scalar('Loss/Policy', policy_loss.item(), outStpCnt)
            writer.add_scalar('Loss/Value', value_loss.item(), outStpCnt)
            #writer.add_scalar('Loss/Total', total_loss.item(), allStpCnt)

            # Log gradient norms
            total_grad_norm = 0
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}/values', param, outStpCnt)
                if param.grad is not None:
                    writer.add_histogram(f'{name}/gradients', param.grad, outStpCnt)
                    param_norm = param.grad.data.norm(2).item()
                    writer.add_scalar(f'GradNorm/{name}', param_norm, outStpCnt)
                    total_grad_norm += param_norm ** 2
            total_grad_norm = math.sqrt(total_grad_norm)
            writer.add_scalar('GradNorm/Total', total_grad_norm, outStpCnt)
            print('tensorboard updated!!')
        '''

        ensure_shared_grads(model, shared_model)

        '''debugging checking if grad is none
        for name, param in shared_model.named_parameters():
            if param.grad is None:
                print(f"Gradient for {name} is None.")
        '''

        '''
        with lock:
            if outStpCnt % 1000 == 0:
                total_grad_norm_shared = 0
                for name, param in shared_model.named_parameters():
                    writer.add_histogram(f'{name}/values', param, outStpCnt)
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        writer.add_histogram(f'{name}/gradients', param.grad, outStpCnt)
                        writer.add_scalar(f'SharedGradNorm/{name}', param_norm, outStpCnt)
                        total_grad_norm_shared += param_norm ** 2
                total_grad_norm_shared = math.sqrt(total_grad_norm_shared)
                writer.add_scalar('SharedGradNorm/Total', total_grad_norm_shared, outStpCnt)
                print('Shared model gradients logged to TensorBoard.')
        '''
        '''
        with lock:
            gradients_match = True
            for param, shared_param in zip(model.parameters(), shared_model.parameters()):
                if shared_param.grad is not None and param.grad is not None:
                    if not torch.allclose(param.grad, shared_param.grad):
                        gradients_match = False
                        print(f"Gradients do not match for parameter: {param}")
                        break
            if gradients_match:
                print("Gradients successfully copied to shared model.")
        '''

        optimizer.step()

        '''

        with lock:
            if outStpCnt % 1000 == 0:
                for name, param in shared_model.named_parameters():
                    writer.add_histogram(f'SharedModelParams/{name}', param.clone().cpu().data.numpy(), outStpCnt)
                print(f"Worker {rank}: Shared model parameters logged.")
        '''

        outStpCnt += 1
