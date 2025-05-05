import os
import time
import gym
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import deque
from gym_super_mario_bros import make as make_mario
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# =================== ENV WRAPPERS ===================

class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayscaleAndResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor()
        ])
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(84, 84), dtype=np.float32)

    def observation(self, obs):
        return self.transform(obs).squeeze(0).numpy()

class StackFrames(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(k, *shp),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames, axis=0), reward, done, info

def init_environment(seed=42):
    env = make_mario("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrames(env, skip=4)
    env = GrayscaleAndResize(env)
    env = StackFrames(env, k=4)
    env.seed(seed)
    return env

# =================== Q-NETWORK ===================

class DuelingCNN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(3136, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.extractor(x)
        x = F.relu(self.fc(x))
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


# =================== REPLAY BUFFER ===================

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def add(self, priority, sample):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = sample
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        diff = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx:
            idx = (idx - 1) // 2
            self.tree[idx] += diff

    def sample(self, value):
        idx = 0
        while True:
            left, right = 2 * idx + 1, 2 * idx + 2
            if left >= len(self.tree): break
            idx = left if value <= self.tree[left] else right
            if value > self.tree[left]:
                value -= self.tree[left]
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]


class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, *args):
        sample = args
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, sample)

    def sample(self, batch_size, beta=0.4):
        segment = self.tree.total() / batch_size
        batch, weights, idxs = [], [], []

        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, p, data = self.tree.sample(s)
            prob = p / self.tree.total()
            weight = (self.tree.size * prob) ** (-beta)
            batch.append(data)
            weights.append(weight)
            idxs.append(idx)

        weights = np.array(weights) / max(weights)
        return list(zip(*batch)), np.array(idxs), weights

    def update(self, idxs, td_errors):
        for idx, error in zip(idxs, td_errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


# =================== AGENT ===================

class DQNAgent:
    def __init__(self, state_shape, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DuelingCNN(state_shape[0], n_actions).to(self.device)
        self.target_net = DuelingCNN(state_shape[0], n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.memory = PrioritizedBuffer(10000)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=5e-5)
        self.gamma = 0.99
        # Epsilon exploration parameters
        self.epsilon_min = 0.05
        self.epsilon_decay = 1e-6
        self.base_epsilon = 0.3           # decays over time
        self.boosted_epsilon = 0.3        # constant during exploration boost
        self.epsilon = self.base_epsilon  # active value used during action selection
        self.batch_size = 128
        self.target_sync = 1000
        self.step = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.q_net.advantage.out_features - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.q_net(state)
            action = q_vals.argmax(dim=1).item()

        # Decay epsilon after every action
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return action
    def train_step(self, beta=0.4):
        if len(self.memory.tree.data) < self.batch_size:
            return
        (states, actions, rewards, next_states, dones), idxs, weights = self.memory.sample(self.batch_size, beta)

        s = torch.from_numpy(np.array(states)).float().to(self.device)
        s_ = torch.from_numpy(np.array(next_states)).float().to(self.device)
        a = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(self.device)
        d = torch.from_numpy(np.array(dones)).float().unsqueeze(1).to(self.device)
        w = torch.from_numpy(np.array(weights)).float().unsqueeze(1).to(self.device)

        q = self.q_net(s).gather(1, a)
        with torch.no_grad():
            next_a = self.q_net(s_).argmax(1, keepdim=True)
            target_q = self.target_net(s_).gather(1, next_a)
            target = r + self.gamma * (1 - d) * target_q

        td_error = target - q
        loss = (F.smooth_l1_loss(q, target, reduction='none') * w).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update(idxs, td_error.detach().cpu().numpy().squeeze())

        if self.step % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.step += 1


# =================== TRAINING LOOP ===================
def train_agent(
    episodes=10000,
    resume_path=None,
    eval_interval=10000,
    warmup_steps=50000,
    revert_threshold=0.95,
):
    env = init_environment()
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    agent.base_epsilon = agent.epsilon
    agent.boosted_epsilon = 0.3
    in_boost_mode = False

    best_reward = float('-inf')
    best_avg_reward = float('-inf')
    best_model_state = None

    episode = 0
    total_reward = 0
    state = env.reset()

    # episode tracking for average score in each interval
    episode_rewards = []
    episode_steps = []
    last_eval_step = 0

    if resume_path and os.path.exists(resume_path):
        print(f"🔁 Resuming training from {resume_path}")
        agent.q_net.load_state_dict(torch.load(resume_path, map_location=agent.device))
        agent.target_net.load_state_dict(agent.q_net.state_dict())

    for _ in range(4):
        action = env.action_space.sample()
        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()

    for step in tqdm(range(1, int(1e7) + 1)):
        agent.epsilon = agent.boosted_epsilon if in_boost_mode else agent.base_epsilon

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.train_step(beta=min(1.0, 0.4 + step / 1e6))
        agent.base_epsilon = max(agent.epsilon_min, agent.base_epsilon - agent.epsilon_decay)

        if best_reward > 0 and total_reward >= 0.95 * best_reward:
            in_boost_mode = True
        else:
            in_boost_mode = False

        if done:
            eval_reward = evaluate_current(agent, episodes=1)
            print(f"\n[Episode {episode}] Reward: {total_reward:.1f} | Best: {best_reward:.1f} | ε: {agent.epsilon:.3f} | Eval Reward: {eval_reward:.1f}")

            # (1) Store best model if this episode's reward is the highest ever
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_model_state = agent.q_net.state_dict()
                torch.save(best_model_state, "best_model.pth")
                print(f"💾 New best model saved with eval reward: {best_reward:.1f}")

            # track reward and step for eval
            episode_rewards.append(eval_reward)
            episode_steps.append(step)

            # (2) every ~eval_interval steps, compute average and maybe revert
            if step - last_eval_step >= eval_interval and step >= warmup_steps:
                # only include episodes finished in this interval
                recent_rewards = [r for s, r in zip(episode_steps, episode_rewards) if s > last_eval_step]
                if recent_rewards:
                    avg_recent = np.mean(recent_rewards)
                    print(f"🧪 Eval window ({len(recent_rewards)} episodes): Avg Reward = {avg_recent:.1f}")
                    if avg_recent < revert_threshold * best_avg_reward and best_model_state is not None:
                        print(f"⚠️ Average reward dropped ({avg_recent:.1f} < {revert_threshold * best_avg_reward:.1f}) — reverting.")
                        agent.q_net.load_state_dict(best_model_state)
                        agent.target_net.load_state_dict(best_model_state)
                        in_boost_mode = True
                        print(f"🔁 Reverted to best model. Boost mode ON.")
                    last_eval_step = step
                    print(f"Average reward is {avg_recent:.1f}")
                    best_avg_reward = max(best_avg_reward, avg_recent)

            # reset
            state = env.reset()
            for _ in range(4):
                action = env.action_space.sample()
                state, _, done, _ = env.step(action)
                if done:
                    state = env.reset()
            total_reward = 0
            episode += 1

        if step % 50000 == 0:
            torch.save(agent.q_net.state_dict(), f"checkpoint_step{step}.pth")

    torch.save(agent.q_net.state_dict(), "rainbow_final.pth")
    env.close()



def evaluate_current(agent_model, episodes=1, render=False):
    from student_agent import Agent
    env = init_environment(seed=1337 + ep)
    agent = Agent(model=agent_model.q_net)

    total_rewards = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            if render:
                env.render()
            # Pass already-stacked frame (shape [4, 84, 84])
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward

        total_rewards.append(ep_reward)
        print(f"[Eval Episode {ep + 1}] Reward: {ep_reward:.1f}")

    env.close()
    return np.mean(total_rewards)





def evaluate_agent(checkpoint_path="rainbow_final.pth", episodes=10, render=False):
    env = init_environment()
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    agent.q_net.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
    agent.q_net.eval()

    total_rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if render:
                env.render()
                time.sleep(0.01)  # add delay for visibility

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                action = agent.q_net(state_tensor).argmax(dim=1).item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"[Eval Episode {ep+1}] Reward: {episode_reward:.1f}")

    avg_reward = np.mean(total_rewards)
    print(f"\n✅ Average reward over {episodes} episodes: {avg_reward:.2f}")
    env.close()

def evaluate_all_checkpoints(start=50000, end=1000000, step=50000, episodes=1, render=False):
    for ckpt_step in range(start, end + 1, step):
        checkpoint_path = f"checkpoint2_step{ckpt_step}.pth"
        print(f"\n🔍 Evaluating checkpoint: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            evaluate_agent(checkpoint_path=checkpoint_path, episodes=episodes, render=render)
        else:
            print(f"⚠️ Checkpoint {checkpoint_path} not found.")

if __name__ == "__main__":
    train_agent(
        episodes=1000000,
        resume_path="checkpoint2_step300000.pth",
        warmup_steps=10000,
        revert_threshold=0.95
    )