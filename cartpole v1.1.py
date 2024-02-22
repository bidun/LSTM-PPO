import gym
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_space, action_space,chkpt_dir='tmp/ppo'):
        super(Agent, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'lstm_ppo')
        
        
        
        #actor
        self.network_actor = nn.Sequential(
            layer_init(nn.Linear(observation_space, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )
        self.lstm_actor = nn.LSTM(512, 512)
        for name, param in self.lstm_actor.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(512, action_space), std=0.01)
        

        #critic
        self.lstm_critc = nn.LSTM(512, 512)
        self.network_critic = nn.Sequential(
            layer_init(nn.Linear(observation_space, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )
        
        for name, param in self.lstm_critc.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        
        self.critic = layer_init(nn.Linear(512, 1), std=1)

        

    def get_states_actor(self, x, lstm_state, done):
        #normalize the input before feed in the lstm is it really hidden? is hidden input not hidden or cell
        hidden = self.network_actor(x / 1.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm_actor.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        
        for h, d in zip(hidden, done):
            #print(h.unsqueeze(0))
            h, lstm_state = self.lstm_actor(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state


    def get_states_critic(self, x, lstm_state, done):
        #normalize the input before feed in the lstm is it really hidden? is hidden input not hidden or cell
        hidden = self.network_critic(x / 1.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm_critc.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        
        for h, d in zip(hidden, done):
            #print(h.unsqueeze(0))
            h, lstm_state = self.lstm_critc(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states_critic(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state_actor, lstm_state_critic, done, action=None):
        hidden_actor, lstm_state_actor = self.get_states_actor(x, lstm_state_actor, done)
        logits = self.actor(hidden_actor)
        
        hidden_critic, lstm_state_critic = self.get_states_critic(x, lstm_state_critic, done)
        

        probs = Categorical(logits=logits)
        #print(probs.probs)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden_critic), lstm_state_actor, lstm_state_critic
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


if __name__ == '__main__':
    
    env = gym.make('CartPole-v1')
    #env = gym.make('BreakoutNoFrameskip-v4')
    #environment_name = "BreakoutNoFrameskip-v4" #Env
    #env = gym.make('CartPole-v1',render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    next_obs = env.reset()
    print(next_obs[0])
    print(env.action_space.n)

    next_obs = torch.Tensor(next_obs[0]).to(device)

    
    n_games = 300

    learning_rate = 2.5e-4
    observation_space = 4
    action_space = env.action_space.n

    num_steps = 5000#128
    gae = True
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.1
    norm_adv = True
    clip_vloss = True
    clip_coef = 0.1
    ent_coef = 0.01
    vf_coef = 1 #0.5
    target_kl = None


    max_grad_norm = 0.5
    update_epochs = 4

    #TODO store obs, action, logprobs, rewards, dones

    

    obs = torch.zeros(0, observation_space).to(device)
    actions = torch.zeros(0).to(device)
    logprobs = torch.zeros(0).to(device)
    rewards = torch.zeros(0).to(device)
    dones = torch.zeros(0).to(device)
    values = torch.zeros(0).to(device)


    agent = Agent(observation_space = observation_space, action_space = action_space).to(device)
    #agent.load_checkpoint()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    global_step = 0
    #next_obs = torch.Tensor([1,1,1,0.3]).to(device)
    next_done = torch.zeros(1).to(device)
    next_lstm_state_actor = (#cotain actor h0 and c0
        torch.zeros(agent.lstm_actor.num_layers, 1, agent.lstm_actor.hidden_size).to(device),
        torch.zeros(agent.lstm_actor.num_layers, 1, agent.lstm_actor.hidden_size).to(device),
    ) 
    
    next_lstm_state_critic = (#cotain actor h0 and c0
        torch.zeros(agent.lstm_critc.num_layers, 1, agent.lstm_critc.hidden_size).to(device),
        torch.zeros(agent.lstm_critc.num_layers, 1, agent.lstm_critc.hidden_size).to(device),
    ) 

    #print("obs:", obs)
    next_obs = env.reset()[0]
    next_obs = torch.Tensor([next_obs]).to(device)
    #print(next_obs)
    #obs = torch.cat((obs, next_obs))
    #print(obs)

    #first value:
    done = 0

    max_score = 0
    current_score = 0

    update_weight = 1
    optimizer.zero_grad()

    num_game = 0

    initial_lstm_state_actor = (next_lstm_state_actor[0].clone(), next_lstm_state_actor[1].clone())

    initial_lstm_state_critic = (next_lstm_state_critic[0].clone(), next_lstm_state_critic[1].clone())

    #observation, action done reward, state_
    
    #game before update
    target_game = 5
    total_game =0

    for i in range(0,10000):
        num_steps = 0
        obs = torch.zeros(0, observation_space).to(device)
        actions = torch.zeros(0).to(device)
        logprobs = torch.zeros(0).to(device)
        rewards = torch.zeros(0).to(device)
        dones = torch.zeros(0).to(device)
        values = torch.zeros(0).to(device)
        #print(num_steps)
        while 1:

            if next_done == 1:
                num_game += 1
                total_game +=1
                #reset done
                next_done = torch.zeros(1).to(device)


                #reset first state of all 0
                next_lstm_state_actor = (#cotain actor h0 and c0
                    torch.zeros(agent.lstm_actor.num_layers, 1, agent.lstm_actor.hidden_size).to(device),
                    torch.zeros(agent.lstm_actor.num_layers, 1, agent.lstm_actor.hidden_size).to(device),
                ) 
                
                next_lstm_state_critic = (#cotain actor h0 and c0
                    torch.zeros(agent.lstm_critc.num_layers, 1, agent.lstm_critc.hidden_size).to(device),
                    torch.zeros(agent.lstm_critc.num_layers, 1, agent.lstm_critc.hidden_size).to(device),
                ) 
                
                next_obs = env.reset()[0]
                next_obs = torch.Tensor([next_obs]).to(device)

                if num_game == target_game:
                    num_game = 0
                    break
            
            num_steps += 1
            obs = torch.cat((obs, next_obs))

        

            with torch.no_grad():
                action, logprob, _, value, next_lstm_state_actor, next_lstm_state_critic = agent.get_action_and_value(next_obs, next_lstm_state_actor, next_lstm_state_critic, next_done)
                values = torch.cat((values, value))

            actions = torch.cat((actions, action))
            logprobs = torch.cat((logprobs, logprob))

            next_obs, reward, done, info, _ = env.step(action.item())
            current_score += reward
            #done = [0]
            #reward = r[action]
            #if done:
            #    print("done")
            
            #print(reward)
            
            reward_tensor = torch.tensor([reward]).to(device)
            rewards = torch.cat((rewards,reward_tensor))
            

            next_obs, next_done = torch.Tensor([next_obs]).to(device), torch.Tensor([done]).to(device)
            dones = torch.cat((dones, next_done))

        # it just finish a game, the current state is the end state, the end state is stored in the final states, the value of the final state is unknown
             
            
        #print(rewards)
            
        #get the next value and get the advantage


        with torch.no_grad():
            if gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if dones[t] == 1:
                        nextnonterminal = 0
                        nextvalues = 0
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

                
                returns = advantages + values.view(-1)
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(num_steps)):
                    if done[t] == 1:
                        nextnonterminal = 1.0 - 1.0
                        next_return = 0
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - values

        if total_game % 50 ==0:
            print("total game:", total_game, "Average socre of last 5 game: ", num_steps/5)
    
        b_obs = obs
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        #need to check
        #clipfracs = []
        #print(obs.shape)
        #print(b_logprobs.shape)
        #print(b_actions.shape)
        #print(b_dones.shape)
        #print(b_advantages.shape)
        #print(b_returns.shape)
        #print(b_values.shape)

        


        for epoch in range(update_epochs):
            _, newlogprob, entropy, newvalue, _, _ = agent.get_action_and_value(
                b_obs,
                initial_lstm_state_actor,
                initial_lstm_state_critic,
                b_dones,
                b_actions.long(),
            )

            
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()


            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                #clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = b_advantages
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns) ** 2
                v_clipped = b_values + torch.clamp(
                    newvalue - b_values,
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss 
            
            critic_loss = v_loss * vf_coef
            
            optimizer.zero_grad()

            loss.backward()
            
            critic_loss.backward()

            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)

            optimizer.step()

            #print("update")
        
        
        if target_kl is not None:
            if approx_kl > target_kl:
                break
