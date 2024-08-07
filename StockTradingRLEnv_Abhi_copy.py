# import random
# import json
# import gymnasium as gym
# from gymnasium import spaces
# import pandas as pd
# import numpy as np
# import datetime as dt
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import PPO

# INITIAL_ACCOUNT_BALANCE = 25000


# # def treasury_bond_daily_return_rate():
# #     r_year = 2.75 / 100  # approximate annual U.S. Treasury bond return rate
# #     return (1 + r_year)**(1 / 365)

# def treasury_bond_daily_return_rate():
#     r_year = 2.75 / 100  # approximate annual U.S. Treasury bond return rate
#     return (1 + r_year)**(1 / 365) - 1

# def generate_price_state(stock_prices, end_index, window_size):
#     '''
#     return a state representation, defined as
#     the adjacent stock price differences after sigmoid function (for the past window_size days up to end_date)
#     note that a state has length window_size, a period has length window_size+1
#     '''
#     start_index = end_index - window_size
#     if start_index >= 0:
#         period = stock_prices[start_index:end_index+1]
#     else:  # if end_index cannot suffice window_size, pad with prices on start_index
#         period = -start_index * [stock_prices[0]] + stock_prices[0:end_index+1]
#     return sigmoid(np.diff(period))


# def generate_portfolio_state(stock_price, balance, num_holding):
#     '''logarithmic values of stock price, portfolio balance, and number of holding stocks'''
#     return [np.log(stock_price), np.log(balance), np.log(num_holding + 1e-6)]


# def generate_combined_state(end_index, window_size, stock_prices, balance, num_holding):
#     '''
#     return a state representation, defined as
#     adjacent stock prices differences after sigmoid function (for the past window_size days up to end_date) plus
#     logarithmic values of stock price at end_date, portfolio balance, and number of holding stocks
#     '''
#     prince_state = generate_price_state(stock_prices, end_index, window_size)
#     portfolio_state = generate_portfolio_state(
#         stock_prices[end_index], balance, num_holding)
#     return np.array([np.concatenate((prince_state, portfolio_state), axis=None)])

# class StockTradingEnv(gym.Env):
#     """A stock trading environment for OpenAI gym"""

#     def __init__(self, df, render_mode=None):
#         super(StockTradingEnv, self).__init__()

#         self.df = df
#         self.render_mode = render_mode
#         self.reward_range = (0, np.inf)

#         # Actions of the format Buy x%, Sell x%, Hold, etc.
#         self.action_space = spaces.Box(
#             low=np.array([0, 0]), high=np.array([2, 1]), dtype=np.float16)

#         # Prices contains the OHLC values for the last five prices
#         self.observation_space = spaces.Box(
#             low=0, high=1, shape=(6, 6), dtype=np.float16)

#     def _next_observation(self):
#         # # Get the stock data points for the last 5 days and scale to between 0-1
#         # frame = np.array([
#         #     self.df.loc[self.current_step: self.current_step +
#         #                 5, 'Open'].values / self.df.loc[:self.current_step, 'Open'].mean(),
#         #     self.df.loc[self.current_step: self.current_step +
#         #                 5, 'High'].values / self.df.loc[:self.current_step, 'High'].mean(),
#         #     self.df.loc[self.current_step: self.current_step +
#         #                 5, 'Low'].values / self.df.loc[:self.current_step, 'Low'].mean(),
#         #     self.df.loc[self.current_step: self.current_step +
#         #                 5, 'Close'].values / self.df.loc[:self.current_step, 'Close'].mean(),
#         #     self.df.loc[self.current_step: self.current_step +
#         #                 5, 'Volume'].values / self.df.loc[:self.current_step, 'Volume'].mean()
#         # ])

#         # obs = np.append(frame, [[
#         #     self.balance / INITIAL_ACCOUNT_BALANCE,
#         #     self.max_net_worth / INITIAL_ACCOUNT_BALANCE,
#         #     self.shares_held /
#         #     self.df.loc[:self.current_step, 'Volume'].mean(),
#         #     self.cost_basis / self.df.loc[:self.current_step, 'Close'].mean(),
#         #     self.total_shares_sold /
#         #     self.df.loc[:self.current_step, 'Volume'].mean(),
#         #     self.total_sales_value / (self.df.loc[:self.current_step, 'Volume'].mean(
#         #     ) * self.df.loc[:self.current_step, 'Close'].mean()),
#         # ]], axis=0)

#         # return obs
#         return generate_combined_state(
#             t, window_size, stock_prices, self.balance, len(self.inventory))

#     def _take_action(self, action):
#         current_price = random.uniform(
#             self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

#         action_type = action
#         # amount = 1

#         if action_type == 0:  # 0=Buy, 1=Sold, 2=Hold
#             total_possible = int(self.balance / current_price)
#             # shares_bought = int(total_possible * amount)
#             shares_bought = min(total_possible, 1)
#             prev_cost = self.cost_basis * self.shares_held
#             additional_cost = shares_bought * current_price

#             self.balance -= additional_cost

#             v1 = prev_cost + additional_cost
#             v2 = self.shares_held + shares_bought
#             self.cost_basis = 0 if v2 == 0 else v1 / v2
#             self.shares_held += shares_bought

#         elif action_type == 2:
#             # shares_sold = int(self.shares_held * amount)
#             shares_sold = 1
#             self.balance += shares_sold * current_price
#             self.shares_held -= shares_sold
#             self.total_shares_sold += shares_sold
#             self.total_sales_value += shares_sold * current_price

#         self.net_worth = self.balance + self.shares_held * current_price

#         if self.net_worth > self.max_net_worth:
#             self.max_net_worth = self.net_worth

#         if self.shares_held == 0:
#             self.cost_basis = 0

#     def step(self, action):
#         self._take_action(action)
#         self.current_step += 1

#         if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
#             self.current_step = 0

#         # self.bond_net_worth *= treasury_bond_daily_return_rate()
#         # reward = self.net_worth-self.bond_net_worth
#         reward = self.net_worth
#         # reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
#         done = self.net_worth <= 0
#         truncated = False  # You can set this to True based on your logic
#         obs = self._next_observation()

#         return obs, reward, done, truncated, {}

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         # self.bond_net_worth = INITIAL_ACCOUNT_BALANCE
#         self.balance = INITIAL_ACCOUNT_BALANCE
#         self.net_worth = INITIAL_ACCOUNT_BALANCE
#         self.max_net_worth = INITIAL_ACCOUNT_BALANCE
#         self.shares_held = 0
#         self.cost_basis = 0
#         self.total_shares_sold = 0
#         self.total_sales_value = 0

#         self.inventory = []
#         self.return_rates = []
#         self.portfolio_values = [INITIAL_ACCOUNT_BALANCE]
#         self.buy_dates = []
#         self.sell_dates = []

#         self.current_step = random.randint(
#             0, len(self.df.loc[:, 'Open'].values) - 6)

#         obs = self._next_observation()
#         return obs, {}  # Return the observation and an empty info dictionary

#     def render(self, mode='human', close=False):
#         if self.render_mode is None or self.render_mode == 'human':
#             profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
#             print(f'Step: {self.current_step}')
#             print(f'Balance: {self.balance}')
#             print(f'Shares held: {
#                   self.shares_held} (Total sold: {self.total_shares_sold})')
#             print(f'Avg cost for held shares: {
#                   self.cost_basis} (Total sales value: {self.total_sales_value})')
#             print(f'Net worth: {
#                   self.net_worth} (Max net worth: {self.max_net_worth})')
#             print(f'Profit: {profit}')
#         else:
#             raise NotImplementedError(
#                 f'Render mode {self.render_mode} is not implemented')


# if (__name__ == '__main__'):
#     df = pd.read_csv('./data/AAPL.csv')
#     df = df.sort_values('Date')
#     df.dropna(inplace=True)
#     df = df.sort_values('Date')
#     df = df.reset_index(drop=True)

#     # The algorithms require a vectorized environment to run
#     env = DummyVecEnv([lambda: StockTradingEnv(df, render_mode='human')])

#     model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=20000)

#     obs = env.reset()
#     for i in range(2000):
#         action, _states = model.predict(obs)
#         obs, rewards, done, truncated, _ = env.step(action)
#         env.render()


# import random
# import gymnasium as gym
# from gymnasium import spaces
# import pandas as pd
# import numpy as np
# from scipy.special import expit as sigmoid
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import PPO

# INITIAL_ACCOUNT_BALANCE = 25000


# def stock_close_prices(df):
#     '''return a list containing stock close prices from a dataframe'''
#     return df['Close'].values.tolist()


# def generate_price_state(stock_prices, end_index, window_size):
#     '''
#     return a state representation, defined as
#     the adjacent stock price differences after sigmoid function (for the past window_size days up to end_date)
#     note that a state has length window_size, a period has length window_size+1
#     '''
#     start_index = end_index - window_size
#     if start_index >= 0:
#         period = stock_prices[start_index:end_index+1]
#     else:  # if end_index cannot suffice window_size, pad with prices on start_index
#         period = -start_index * [stock_prices[0]] + stock_prices[0:end_index+1]
#     return sigmoid(np.diff(period))


# def generate_portfolio_state(stock_price, balance, num_holding):
#     '''logarithmic values of stock price, portfolio balance, and number of holding stocks'''
#     return [np.log(stock_price), np.log(balance), np.log(num_holding + 1e-6)]


# def generate_combined_state(end_index, window_size, stock_prices, balance, num_holding):
#     '''
#     return a state representation, defined as
#     adjacent stock prices differences after sigmoid function (for the past window_size days up to end_date) plus
#     logarithmic values of stock price at end_date, portfolio balance, and number of holding stocks
#     '''
#     prince_state = generate_price_state(stock_prices, end_index, window_size)
#     portfolio_state = generate_portfolio_state(
#         stock_prices[end_index], balance, num_holding)
#     return np.array([np.concatenate((prince_state, portfolio_state), axis=None)])


import random
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import torch
from scipy.special import expit as sigmoid
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import numpy as np

INITIAL_ACCOUNT_BALANCE = 25000

reward = 0


def stock_close_prices(df):
    '''return a list containing stock close prices from a dataframe'''
    return df['Close'].values.tolist()


def generate_price_state(stock_prices, end_index, window_size):
    '''
    return a state representation, defined as
    the adjacent stock price differences after sigmoid function (for the past window_size days up to end_date)
    note that a state has length window_size, a period has length window_size+1
    '''
    start_index = end_index - window_size
    if start_index >= 0:
        period = stock_prices[start_index:end_index+1]
    else:  # if end_index cannot suffice window_size, pad with prices on start_index
        period = -start_index * [stock_prices[0]] + stock_prices[0:end_index+1]
    return sigmoid(torch.diff(torch.tensor(period))).clone().detach()
    # return torch.tensor(sigmoid(torch.diff(torch.tensor(period))), dtype=torch.float32)


def generate_portfolio_state(stock_price, balance, num_holding):
    '''logarithmic values of stock price, portfolio balance, and number of holding stocks'''
    return torch.tensor([torch.log(torch.tensor(stock_price)),
                        torch.log(torch.tensor(balance)),
                        torch.log(torch.tensor(num_holding) + 1e-6)], dtype=torch.float32)


def generate_combined_state(end_index, window_size, stock_prices, balance, num_holding):
    '''
    return a state representation, defined as
    adjacent stock prices differences after sigmoid function (for the past window_size days up to end_date) plus
    logarithmic values of stock price at end_date, portfolio balance, and number of holding stocks
    '''
    price_state = generate_price_state(stock_prices, end_index, window_size)
    portfolio_state = generate_portfolio_state(
        stock_prices[end_index], balance, num_holding)
    return torch.cat((price_state, portfolio_state))


def treasury_bond_daily_return_rate():
    r_year = 2.75 / 100  # approximate annual U.S. Treasury bond return rate
    return (1 + r_year)**(1 / 365) - 1


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    def __init__(self, df, render_mode=None):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.render_mode = render_mode
        self.reward_range = (0, float('inf'))
        self.stock_prices = stock_close_prices(df)
        self.inventory = []
        self.window_size = 5

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([2]), dtype=np.float16)

        # Observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, self.window_size + 3), dtype=np.float16)

    def _next_observation(self):
        return generate_combined_state(
            self.current_step, self.window_size, self.stock_prices, self.balance, self.shares_held)

    def hold(self, t, q_values):
        # encourage selling for profit and liquidity
        q_values = q_values.detach().numpy()[0]
        next_probable_action = np.argsort(q_values)[1]
        if next_probable_action == 2 and len(self.inventory) > 0:
            max_profit = self.stock_prices[t] - min(self.inventory)
            if max_profit > 0:
                self.sell(t)
                return 'Hold'

    def buy(self, t):
        if self.balance > self.stock_prices[t]:
            self.balance -= self.stock_prices[t]
            self.inventory.append(self.stock_prices[t])
            return 'Buy: ${:.2f}'.format(self.stock_prices[t])

    def sell(self, t):
        if len(self.inventory) > 0:
            self.balance += self.stock_prices[t]
            bought_price = self.inventory.pop(0)
            profit = self.stock_prices[t] - bought_price
            global reward
            reward += profit
            return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(self.stock_prices[t], profit)

    def step(self, action, q_values):
        # reward = self._take_action(action, q_values)
        reward = 0
        if action == 0:  # hold
            execution_result = self.hold(self.current_step, q_values)
        elif action == 1:  # buy
            execution_result = self.buy(self.current_step)
        elif action == 2:  # sell
            execution_result = self.sell(self.current_step)

        if execution_result is None:
            reward -= treasury_bond_daily_return_rate() * self.balance  # missing opportunity
        # else:
        #     if isinstance(execution_result, tuple):  # if execution_result is 'Hold'
        #         actions = execution_result[1]
        #         execution_result = execution_result[0]

        self.net_worth = len(
            self.inventory) * self.stock_prices[self.current_step] + self.balance

        unrealized_profit = self.net_worth - self.initial_portfolio_value
        reward += unrealized_profit
        done = self.net_worth <= 0

        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'Open'].values) - 1:
            self.current_step = 0

        obs = self._next_observation()

        return obs, q_values, reward, done

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.initial_portfolio_value = INITIAL_ACCOUNT_BALANCE
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.inventory = []
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 1)

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
