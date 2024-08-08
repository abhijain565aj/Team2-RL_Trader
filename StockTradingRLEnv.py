import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import datetime as dt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

INITIAL_ACCOUNT_BALANCE = 20000


def treasury_bond_daily_return_rate():
    r_year = 2.75 / 100  # approximate annual U.S. Treasury bond return rate
    return (1 + r_year)**(1 / 365)


def xmodx(x):
    return x*abs(x)


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    def __init__(self, df, render_mode=None):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.render_mode = render_mode
        self.reward_range = (0, np.inf)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float16)

        # Prices contains the OHLC values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        # Ensure there are enough previous days to look back
        frame_start = max(0, self.current_step - 5)

        # Collect the last 5 days of data, padding with zeros if necessary
        frame = np.array([
            self.df.loc[frame_start:self.current_step, 'Open'].values,
            self.df.loc[frame_start:self.current_step, 'High'].values,
            self.df.loc[frame_start:self.current_step, 'Low'].values,
            self.df.loc[frame_start:self.current_step, 'Close'].values,
            self.df.loc[frame_start:self.current_step, 'Volume'].values
        ])

        # Pad with zeros if the number of days is less than 5
        if frame.shape[1] < 6:
            padding = 6 - frame.shape[1]
            frame = np.pad(frame, ((0, 0), (padding, 0)),
                           'constant', constant_values=0)

        # Normalize by the mean value over the observed period
        frame = frame / frame.mean(axis=1, keepdims=True)

        obs = np.append(frame, [[
            self.balance / INITIAL_ACCOUNT_BALANCE,
            self.max_net_worth / INITIAL_ACCOUNT_BALANCE,
            self.shares_held / self.df.loc[frame_start:self.current_step, 'Volume'].mean(
            ) if self.df.loc[frame_start:self.current_step, 'Volume'].mean() != 0 else 1,
            self.cost_basis / self.df.loc[frame_start:self.current_step, 'Close'].mean(
            ) if self.df.loc[frame_start:self.current_step, 'Close'].mean() != 0 else 1,
            self.total_shares_sold / self.df.loc[frame_start:self.current_step, 'Volume'].mean(
            ) if self.df.loc[frame_start:self.current_step, 'Volume'].mean() != 0 else 1,
            self.total_sales_value / (self.df.loc[frame_start:self.current_step, 'Volume'].mean() * self.df.loc[frame_start:self.current_step, 'Close'].mean()) if (
                self.df.loc[frame_start:self.current_step, 'Volume'].mean() != 0 and self.df.loc[frame_start:self.current_step, 'Close'].mean() != 0) else 1,
        ]], axis=0)  # Transpose to match the shape
        return obs

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        self.allstocks = (INITIAL_ACCOUNT_BALANCE *
                          current_price)/self.df.loc[0, "Open"]

        action_type = action
        # amount = 1

        if action_type == 0:  # 0=Buy, 1=Sold, 2=Hold
            if (self.balance > current_price):
                shares_bought = 1
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price

                self.balance -= additional_cost

                v1 = prev_cost + additional_cost
                v2 = self.shares_held + shares_bought
                self.cost_basis = 0 if v2 == 0 else v1 / v2
                self.shares_held += shares_bought

        elif action_type == 1:
            if (self.shares_held > 0):
                shares_sold = 1
                self.balance += shares_sold * current_price
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        # print("action:", action, "networth:", self.net_worth, "balance:", self.balance,
            #   "shares_held:", self.shares_held, "current_price:", current_price)

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
        return

    def step(self, action):
        before_net_worth = self.net_worth
        
        #delta_loss
        last_loss = self.missed_loss
        self.missed_loss = self.net_worth - self.allstocks
        
        self._take_action(action)
        
        pnl = self.net_worth - self.initial_net_worth
        
        drawdown = self.max_net_worth - self.net_worth #drop from the max_networth
        
        reward = self.net_worth-before_net_worth
        reward += self.missed_loss - last_loss - 0.01*(drawdown)

        # print("networth:", self.net_worth, "reward:", reward)
        
        self.current_step += 1
        done = self.net_worth <= 0
        obs = self._next_observation()

        return obs, reward, done

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.initial_net_worth = INITIAL_ACCOUNT_BALANCE
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.allstocks = INITIAL_ACCOUNT_BALANCE
        self.initial_price = self.df.loc[0, "Open"]
        self.maxallstocks = self.allstocks
        self.missed_loss = 0

        obs = self._next_observation()
        return obs  # Return the observation and an empty info dictionary

    def render(self, mode='human', close=False):
        if self.render_mode is None or self.render_mode == 'human':
            profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Shares held: {
                  self.shares_held} (Total sold: {self.total_shares_sold})')
            print(f'Avg cost for held shares: {
                  self.cost_basis} (Total sales value: {self.total_sales_value})')
            print(f'Net worth: {
                  self.net_worth} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
        else:
            raise NotImplementedError(
                f'Render mode {self.render_mode} is not implemented')


if (__name__ == '__main__'):
    df = pd.read_csv('./data/AAPL.csv')
    df = df.sort_values('Date')
    df.dropna(inplace=True)
    df = df.sort_values('Date')
    df = df.reset_index(drop=True)

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df, render_mode='human')])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done = env.step(action)
        env.render()
