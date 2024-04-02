## Sys and warnings
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import datetime


## Common imports
import matplotlib.pyplot as plt
import itertools
import matplotlib
import numpy as np
import pandas as pd

## Custom imports
from stable_baselines3 import DDPG
from brainLib.customEnv import StockTradingEnv

## FinRL
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl import config
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

sys.path.append("../FinRL")

class GenericTrader:
    def __init__(self):
        pass

    def load_agent(self,agent_path,agent_type):
        if   agent_type.lower()=="a2c":
            from stable_baselines3 import A2C
            return A2C.load(agent_path)
            
        elif agent_type.lower()=="ddpg": 
            from stable_baselines3 import DDPG
            return DDPG.load(agent_path)
            
        elif agent_type.lower()=="ppo": 
            from stable_baselines3 import PPO
            return PPO.load(agent_path)
            
        elif agent_type.lower()=="sac": 
            from stable_baselines3 import SAC
            return SAC.load(agent_path)
            
        elif agent_type.lower()=="td3": 
            from stable_baselines3 import TD3
            return TD3.load(agent_path) 
        else:
            raise ValueError("Must inform a valid agent_type :['a2c','ddpg','ppo','sac','td3']")
               

    def consolidate_by_day(self, action_memory="default"):
        if action_memory=="default":
            action_memory = pd.read_csv("results/save_action_memory.csv")
        data = action_memory.copy()
        data_columns = data.columns
        data['date'] = pd.to_datetime(data['date']) 
    
        data.sort_values('date', inplace=True)  
        
        stock_columns = [col for col in data.columns if col != 'date']
        
        for col in stock_columns:
            data[f'total_{col}'] = data[col].cumsum()
    
        result_cols = ['date'] + [f'total_{col}' for col in stock_columns]
        
        
        new_data = data[result_cols].copy()
        new_data.columns = data_columns
        return new_data
        
   
    def get_trading_data(self,path, start_date, end_date, symbol, new =False ):
        processed_full = pd.read_csv(path, index_col=0)
        
      
        ## Filter by simbol
        if len(symbol)>0:
            processed_full =processed_full[processed_full['tic'] == symbol.upper()]
            
    ## Filter by date 
        date_filter = (processed_full['date'] >= start_date) & (processed_full['date'] <= end_date)
        processed_full=processed_full[date_filter]

        ## Format the index to associate all the rows with the day number 
        ## (starting from zero (first date) to the position of the last date)
        processed_full['Unnamed: 0'] = processed_full['date'].rank(method='dense').astype(int) - 1
        processed_full.index=processed_full['Unnamed: 0']
        
        processed_full.to_csv("processed_full.csv")
        return processed_full
          

    def create_environment(self,trade_limit,initial_amount,dataset, start_date, end_date,resume_session,user, symbol, turbulence_limit= 70, turbulence_quantile=0.996, vix_quantile= 0.996):
        
        stock_dimension = len(dataset.tic.unique())
        state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension
        
        # Create a session for the user+symbol
        session_id = user+symbol
        
         
        if (resume_session):         
            _day , session_amount, session_shares = self.get_last_session(session_id)

            if _day !="zero":
                
                initial_amount  = session_amount
                num_stock_shares = session_shares
            else:
                 num_stock_shares = [0] * stock_dimension
                
        else:    
            num_stock_shares = [0] * stock_dimension
                    
        env_kwargs = {
            "hmax": int(trade_limit),
            "initial_amount"     : float(initial_amount),
            "num_stock_shares"   : num_stock_shares,
            "buy_cost_pct"       : buy_cost_list,
            "sell_cost_pct"      : sell_cost_list,
            "state_space"        : state_space,
            "stock_dim"          : stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space"       : stock_dimension,
            "reward_scaling"     : 1e-4,
            "session_id"         : session_id
                     }
    
        return StockTradingEnv(turbulence_threshold = 70,risk_indicator_col='vix',df = dataset, **env_kwargs)

    
    def run_trade(self,agent, environment):
        
        predictor = DRLAgent(env=environment)
        acct,acts = predictor.DRL_prediction(model=agent, environment=environment)
        new_env   = predictor.env
        
        return acct, acts

    def start_simulation(self,**kwargs):
  
        trader_agent   = self.load_agent(kwargs["agent_path"],
                                         kwargs["agent_type"])    
        
        data      = self.get_trading_data(kwargs["data_path"],
                                          kwargs["start_date"], 
                                          kwargs["end_date"],
                                          kwargs["symbol"],
                                          new =False)
        
        if kwargs["env"] =="":
            env = self.create_environment(kwargs["trade_limit"],
                                          kwargs["initial_amount"],
                                          data, 
                                          kwargs["start_date"],
                                          kwargs["end_date"],
                                          kwargs["resume_session"],
                                          kwargs["user"],
                                          kwargs["symbol"])
        else:
            env=kwargs["env"]
        

        account, actions = self.run_trade(trader_agent,env)
        return  account, actions, env  

    def get_last_session(self,session_id):
        import configparser

        ini_file = configparser.ConfigParser()
        ini_file.read('data/session_memory.ini')
        
        try:
            last_shares =(ini_file[session_id]['shares']).strip("[]").replace("'", "").split(", ")
            

            
            return ini_file[session_id]['day'],float(ini_file[session_id]['money']), [float(share) for share in last_shares]
        
        except Exception as e:
            print("-------------------------")
            print("no session found for this bot")
            print("error:", e)
            print("-------------------------")
            
            return "zero" ,0,0

