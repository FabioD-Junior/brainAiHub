## Libs
from flask import Flask, jsonify, request
import pandas as pd  
import numpy as np  
from sklearn.feature_extraction.text import CountVectorizer   
from scipy.spatial.distance import cosine  
import difflib   
from datetime import datetime

## Config
app = Flask(__name__)

dow_tickers = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 
                             'CSCO', 'CVX', 'DIS','GS', 'HD', 'HON', 'IBM', 
                             'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM','MRK', 
                             'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']


## Endpoints 
@app.route('/chatbot/ask', methods=['GET'])
def ask():
    ## Autentication
    auth_header = request.headers.get('Authorization')
    
    token       = auth_header.split(" ")[1] if auth_header and auth_header.startswith("Bearer ") else None
    user_message = request.args.get('userMessage')
    
    if not token or token != 'eyJhbGciOiJIUd2VyIn0.NDFAApMqRBwacpLumnyC_p7IWWmmWEFmXJIVkRoIA-I':
        return {"message": "Invalid or missing token"}, 401

    kb = pd.read_csv("kb_001.csv")
    
    resp = get_best_response(user_message,kb)
    
    if resp in function_map:
        return jsonify({'botmessage': function_map[resp](user_message) })
        
    return jsonify({'botmessage': resp})
    
## Functions
### Chatbot Main Functios

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
    
    ticker_kb = pd.read_csv("company_info_2024-03-19.csv")
    ticker_kb = ticker_kb[ticker_kb["symbol"]==ticker.upper()]
    
    ## Response variable 
    response = "" 
    
    # Header
    response = ":::::::: Report for the Symbol " + ticker + "::::::::"+ crlf 
    
    # Intro
    response += "::: About the company :::"+ crlf 
    response += ticker_kb["intro"].iloc[0] + crlf + crlf
  
    # News & Sentiment
    response += "::: This is a summary of the most recent related to the " + ticker.upper() + "symbol. We also present a sentiment analysis of the news based on natural language processing, to try to assist in its evaluation. ::: "+ crlf
    
    news      =  ast.literal_eval(ticker_kb["news"].iloc[0])
    sentiment = ast.literal_eval(ticker_kb["sentiment"].iloc[0])
    
    for item in range(len(news)):  
        response += news[item] + crlf
        response += "Sentiment polarity :" + str(sentiment[item][0]) + " - tendency towards :" + sentiment[item][1] + crlf
        
    response += crlf
    
    # Metrics
    response += "::: Recent Metrics for " + ticker + " :::"

    metrics = ast.literal_eval(ticker_kb["metrics"].iloc[0])
 
    for metric in metrics.keys():
       response +=  metric + ":" + str(metrics[metric]) + crlf
          
    return response


### Auxiliary functions


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
        

#### Function Mapping
# Maps custom functions. 
# The key Must be associated with the responses
function_map = {
    "func_time"   : reply_time,
    "func_weather": reply_weather,
    "func_report" : reply_company_info 
}



## Only for runing locally
if __name__ == '__main__':
    app.run(debug=True)
