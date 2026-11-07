import gymnasium as gym
import ale_py
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_frame(frame):
    img = frame[25:201, ::] 
    img = img.max(axis=2)
    img = np.maximum(img[:-1:2, ::2], img[1::2, ::2]) 
    img = img.astype(np.float32) / 255.0
    return img

def stack_frames(stacked_frames, frame, new):
    frame = preprocess_frame(frame) 
    if new:
        stacked_frames = deque([frame for _ in range(4)],maxlen=4,)
    else:
        stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames

class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()
        
        self.common_conv = nn.Sequential(
            #input size is 4, since 4 stacked frames
            nn.Conv2d(4, 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        #input: 4 * 88 * 80
        #conv1: 32 * 21 * 19
        #conv2: 128 * 9 * 8
        #Conv3: 32 * 7 * 6 = 1344
        flattened_size = 32 * 7 * 6

        #actor head
        self.actor = nn.Sequential(
            nn.Linear(flattened_size, 4 * flattened_size),
            nn.ReLU(),
            nn.Linear(4 * flattened_size, action_size),
            nn.Softmax(dim=-1)
        )

        #critic head
        self.critic = nn.Sequential(
            nn.Linear(flattened_size, 4 * flattened_size),
            nn.ReLU(),
            nn.Linear(4 * flattened_size, 1),
        )

    def forward(self, x):
        x = self.common_conv(x)
        x = x.view(x.size(0), -1)
        
        return self.actor(x), self.critic(x)

class A2CAgent:
    def __init__(self, action_size):
        self.model = ActorCritic(action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002)
        self.gamma = 0.99 ** 4

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs, value = self.model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), value, m.entropy()

    def train(self, rewards, log_probs, values, next_value, entropies):
        returns = []
        R = next_value
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns).to(device).view(-1) 
        log_probs = torch.stack(log_probs).view(-1)             
        values = torch.stack(values).view(-1)                   
        entropies = torch.stack(entropies).view(-1)             
        
        advantage = returns - values.detach()
        actor_loss = -(log_probs * advantage).mean()

        critic_loss = nn.functional.mse_loss(values, returns)

        entropy_loss = entropies.mean()

        total_loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    env = gym.make('ALE/SpaceInvaders-v5', render_mode="rgb_array")
    agent = A2CAgent(env.action_space.n)
    model_dir = "models"
    source_model_path = os.path.join(model_dir, "bestmodel4.pth")
    model_path = os.path.join(model_dir, "bestmodel5.pth")
    start_episode = 0

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    agent.model.load_state_dict(torch.load(source_model_path, map_location=device))

    episodes = 1000000
    reward_history = []
    recent_rewards = deque(maxlen=100)
    high_score = 0
    
    for e in range(start_episode, episodes):
        raw_state, _ = env.reset()
        stacked_frames = deque(maxlen=8)
        state, stacked_frames = stack_frames(stacked_frames, raw_state, True)
        
        log_probs = []
        values = []
        rewards = []
        entropies = []
        total_reward = 0
        recording_frames = []
        prev_lives = 3
        done = False
        steps = 1
        mx = 0
        while not done:
            action, log_prob, value, entropy = agent.act(state)
            decision_reward = 0.0
            done = False

            for repeat_index in range(4):
                next_frame, frame_reward, terminated, truncated, info = env.step(action)
                recording_frames.append(env.render())
                total_reward += frame_reward
                done = terminated or truncated

                current_lives = info.get('lives', prev_lives)
                if current_lives < prev_lives:
                    prev_lives = current_lives
                    shaped_reward = -5.0
                elif frame_reward > 0:
                    shaped_reward = 1.0
                elif frame_reward == 0:
                    shaped_reward = 0.001
                else:
                    shaped_reward = frame_reward

            next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(decision_reward)
            entropies.append(entropy)

            steps += 1
            state = next_state
            #update every 64 steps
            if steps % 64 == 0 or done:
                with torch.no_grad():
                    _, _, next_v, _ = agent.act(next_state)
                final_v = 0 if terminated else next_v.item()
                agent.train(rewards, log_probs, values, final_v, entropies)
                avg_entropy = torch.stack(entropies).mean().item()
                print(f"Entropy:     {avg_entropy:.4f}")
                log_probs, values, rewards, entropies = [], [], [], []
            
        reward_history.append(total_reward)
        recent_rewards.append(total_reward)
        average = sum(recent_rewards) / len(recent_rewards)
        if total_reward > high_score:
            print("New High Score!")
            high_score = total_reward
            with open("highscore.txt", "w") as file:
                file.write(str(high_score))
            os.makedirs("recordings", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join("recordings", f"high_score_episode_{e + 1}_{timestamp}.mp4")
            height, width = recording_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (width, height),
            )
            for frame in recording_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"Saved high-score run recording to {video_path}")
        print(f"Episode: {e+1}, Score: {total_reward}, {len(recent_rewards)}-episode average: {average:.2f}")

        if (e + 1) % 10 == 0:
            torch.save(agent.model.state_dict(), model_path)
            print(f"Saved training checkpoint to {model_path}")

    env.close()
