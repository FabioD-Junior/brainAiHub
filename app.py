#--------------------------------------------------------------------------------------------------------#
#                                           Main API                                                     #
#--------------------------------------------------------------------------------------------------------#
# It is an API for Academic Purposes, we do not aim to keep it as "production" or for "bussines purposes"#
# Version : 3.0                                                                                          #
# Source Code with adaptations and "downgrades" in order to be able to run on AWS free tier              #
# Author : Fabio Duarte Junior - fabiojr@skiff.com                                                       #
# Obs: Despite being a flask application, it runs in the master environment through Gunicorn and an Nginx# 
# layer This version includes simulation endpoints. According to the consolidation of APIs carried out in#
# the final project.
#--------------------------------------------------------------------------------------------------------#

## Libs for the main simulation api
from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd

## Libs for the AiHub Chatbot
import numpy as np  
from sklearn.feature_extraction.text import CountVectorizer   
from scipy.spatial.distance import cosine  
import difflib   
from datetime import datetime


## Libs for RAG
import cohere

# ..:: Config
app = Flask(__name__)
CORS(app)

# ..:: Endpoints
## :: Login 
@app.route('/auth/login', methods=['POST'])
def login():
    username = request.headers.get('username')
    secret = request.headers.get('secret')
    if username and secret:
        return jsonify(token='')## TODO : Implement API authentication using JWT from the main API.
    else:
        return {"message": "Wrong Credentials. Try Again"}, 400

## :: BOT 
@app.route('/bot/simulate', methods=['POST'])
def simulate():
    ## Autentication
    auth_header = request.headers.get('Authorization')
    token = auth_header.split(" ")[1] if auth_header and auth_header.startswith("Bearer ") else None
    if not token or token != '<token>': ## TODO : Implement API authentication using JWT from the main API.
        return {"message": "Invalid or missing token"}, 401

    ## Parse JSON from POST request
    data = request.json
    
    ## Running simulation with data from POST request
    asset_history, current_balance, asset_value = run_simulation(data.get('agent_path'),
                                                                 data.get('agent_type'),
                                                                 data.get('data_path'),
                                                                 data.get('trade_limit'),
                                                                 data.get('buy_upper_limit'),
                                                                 data.get('sell_upper_limit'),
                                                                 data.get('initial_amount'),
                                                                 data.get('start_date'),
                                                                 data.get('env'),
                                                                 data.get('end_date'),
                                                                 data.get('symbol'),
                                                                 data.get('user'),
                                                                 data.get('resume_session'),
                                                                 data.get('orientation')
                                                                )
        
    ## Assuming 'run_simulation' and reading 'save_action_memory.csv' remains unchanged
    assets = pd.read_csv("save_action_memory.csv")
    response_data = {
        'asset_history'  : asset_history,
        'current_balance': current_balance,
        'asset_value'    : asset_value,
        'sharpe_ratio'   : 0  # Assuming calculation or retrieval of sharpe_ratio happens here or remains constant
    }
    
    return jsonify(response_data)
    
## :: Bot Auxiliary functions

def run_simulation(agent_path,agent_type,data_path,                    
                   trade_limit,buy_upper_limit, sell_upper_limit,       
                   initial_amount,start_date,env,                
                   end_date,symbol,user, resume_session,orientation):
                   
    from brainLib.brainTrader import GenericTrader
    import pandas as pd

    trader = GenericTrader()
      
    # All Tickers 
    if symbol =="Dow30":
        symbol =  ""
        
    simulation_args =  {"agent_path"    : "agents/"+agent_path+".mdl",
                        "agent_type"    : agent_type,
                        "data_path"     : data_path,
                        "trade_limit"   : trade_limit,
                        "initial_amount": str(initial_amount),
                        "start_date"    : start_date,
                        "env"           : "",
                        "end_date"      : end_date,
                        "symbol"        : symbol,
                        "user"          : user,
                        "resume_session": False
                       }
    print(simulation_args)
    account, actions,env = trader.start_simulation(**simulation_args)  
    

    
    memory        = pd.read_csv("results/state_memory.csv")
    #data          = pd.read_csv("results/asset_memory.csv")
    account_info  = pd.read_csv("results/account_value.csv")
    balance       = memory["money"].iloc[-1]
    account_value = account_info["account_value"].iloc[-1] 
    
    
    return memory.drop(memory.index[0]).to_json(orient=orientation), balance, account_value


#--------------------------------------------------------------------------------------------------------#
#                                           AiHub Chatbot
#--------------------------------------------------------------------------------------------------------#

## ..:: Config
dow_tickers = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 
                             'CSCO', 'CVX', 'DIS','GS', 'HD', 'HON', 'IBM', 
                             'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM','MRK', 
                             'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']


## ..:: Endpoints 
@app.route('/chatbot/ask', methods=['POST'])  # Changed to use the POST method by preference of our front-end team
def ask():
    ## Authentication
    auth_header = request.headers.get('Authorization')
    token = auth_header.split(" ")[1] if auth_header and auth_header.startswith("Bearer ") else None
    
    # Accessing the user message from the POST request body
    # Assuming the data is sent as JSON
    data = request.json
    user_message = data.get('userMessage') if data else None
    
    if not token or token != 'eyJhbGciOiJIUd2VyIn0-I': ## TODO : Implement API authentication using JWT from the main API.
        return {"message": "Invalid or missing token"}, 401

    kb = pd.read_csv("kb_001.csv")
    
    resp = get_best_response(user_message, kb)
    
    if resp in function_map:
        botmessage = function_map[resp](user_message)
    else:
        botmessage = final_response(user_message, resp, 2)
    
    return jsonify({'botmessage': botmessage})


## ..:: Functions
### :: Chatbot Main Functios

def get_best_response(user_message, data):
    """
    Calculates ensemble similarity scores between a user message and a list of messages.
    
    This function combines cosine similarity and Levenshtein distance to compute an
    ensemble similarity score for each message in the list compared to the user's message.
    
    Parameters:
    - user_message (str): The message input by the user.
    - messages (list of str): A list of messages to compare with the user message.
    
    Returns:
    - list of float: A list of ensemble similarity scores corresponding to each message.
    """
    user_message= company_to_symbol(user_message)
    word_list = user_message.split()
    
    generic_message = ["SYMBOL" if word.upper() in dow_tickers else word for word in word_list]
   
    user_message = ' '.join(generic_message)    
    
    
    messages = data['message'].tolist()
    # Initialize the CountVectorizer and fit it to the combined list of messages
    vectorizer = CountVectorizer().fit(messages + [user_message])
    
    # Transform the messages into vectors
    messages_vector = vectorizer.transform(messages).toarray()
    user_vector = vectorizer.transform([user_message]).toarray()[0]
    
    ensemble_scores = []  # Initialize a list to store ensemble scores
    
    # Calculate ensemble score for each message
    for message_vector, message in zip(messages_vector, messages):
        # Compute cosine similarity and normalize it to [0, 1]
        cosine_sim = 1 - cosine(user_vector, message_vector)
        
        # Compute Levenshtein similarity
        levenshtein_sim = difflib.SequenceMatcher(None, user_message, message).ratio()
        
        # Calculate mean of cosine and Levenshtein similarities as ensemble score
        ensemble_score = np.mean([cosine_sim, levenshtein_sim])
        
        ensemble_scores.append(ensemble_score)  # Append ensemble score to the list


    # Finding the best match
    best_score = max(ensemble_scores)
    threshold = 0.5  # Similarity threshold
    
    if best_score < threshold:
        best_response =  "I'm sorry, I didn't understand that. Could you please rephrase?"
    else:
        best_message_index = ensemble_scores.index(best_score)
        best_response = data['response'].iloc[best_message_index]
      


    return best_response


def generate_ticker_report(ticker,crlf="<br>"):
    
    if ticker not in dow_tickers:
        
        return "Sorry, unfortunately we don't have information about this ticker. If you are entering the company name, please try entering the ticker name"
 
    import ast
    
    ticker_kb = pd.read_csv("company_info_2024-04-01.csv")
    ticker_kb = ticker_kb[ticker_kb["symbol"]==ticker.upper()]
  
    info = ticker_kb["intro"].iloc[0]
  
    eval_news      =  ast.literal_eval(ticker_kb["news"].iloc[0])
    
    for item in range(len(eval_news)):  
        news = item
        break
        
    # Metrics
    metric_dict = ast.literal_eval(ticker_kb["metrics"].iloc[0])
    metrics = ""
    for item in metric_dict.keys():
       metrics +=  item + ":" + str(metric_dict[item]) + "<br>"

    news = final_response("" , news, 0)
    info = final_response("", info, 1)

    response = f'{info}. {news}.<br><br> :::: Indicators & Metrics ::::<br><br> {metrics}'
    return response


### :: Auxiliary functions

def get_current_date_time():
    """
    Returns the current date and time in the format of "YYYY-MM-DD HH:MM:SS".
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
def company_to_symbol(message):
    companies = {
    'Apple': 'AAPL', 'Amgen': 'AMGN', 'Amex': 'AXP', 'Boeing': 'BA','Caterpillar': 'CAT', 'salesforce': 'CRM', 'Cisco': 'CSCO', 'Chevron': 'CVX',
    'Disney': 'DIS', 'Goldman': 'GS', 'Depot': 'HD', 'Honeywell': 'HON', 'Intel': 'INTC', 'JPMorgan': 'JPM', 'McDonald': 'MCD', '3M': 'MMM',
    'Merck': 'MRK', 'Microsoft': 'MSFT', 'NIKE': 'NKE', 'UnitedHealth': 'UNH','Visa': 'V', 'Verizon': 'VZ', 'Walgreens': 'WBA', 'Walmart': 'WMT'
    }
    message = message.upper()
    for company, ticker in companies.items():
        message = message.replace(company.upper(), ticker.upper())
    
    return message
def reply_time(message):
    return "It is about : " +  str(get_current_date_time())
    
def reply_weather(message, filler=""):
    weather_list =["good","bad","sunny","rainy","cloudy","snowy"]
    return "I'll guess it might be a " + random.choice(weather_list) +" weather today.(but it is just a guess"

def reply_company_info(message):

    ## Look for the tiker
    new_message = company_to_symbol(message)
    
    for word in new_message.split():
        print("|-> word : ", word) 
        if word.upper() in dow_tickers:
            break
               
    return generate_ticker_report(word)
        

#### :: Function Mapping
# ..::  Maps custom functions. 
## ::   The key Must be associated with the responses
function_map = {
    "func_time"   : reply_time,
    "func_weather": reply_weather,
    "func_report" : reply_company_info 
}

def final_response(message="",response="", prompt_type=0):
    prompts =[f'Summarize an opinion on market sentiment about it by paraphrasing the text in a short and succinct way{response}',
              f'Summarize the information about the following company. Company Info: "{response}"',
              f'''Please, give a more complete response for the question:"{message}". 
              Refrasing the following example of simple response :"{response}". 
              Give only the new response for the question, no additional information or instruction.'''
                ]
    
    ## Connect to Cohere API
    co = cohere.Client('<api key>')

    ## Use the Generation API to enhance our response
    generation = co.generate(
        model='command',  
        prompt=prompts[prompt_type],
        max_tokens=100,  
        temperature=0.5   
        )
    
    ## Transform the output into a dictionary 
    dict_return = generation.dict()

    ## Return the enhanced text
    return dict_return['generations'][0]['text']


# ..:::::: Run! :::::..
if __name__ == '__main__':
    app.run(debug=True)
