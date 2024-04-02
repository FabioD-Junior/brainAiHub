import pandas as pd
import sqlite3
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

class FirestoreDB:
    def __init__(self):
        ## Firebase secret 
        cred = credentials.Certificate("secret/aip-trading-bot-firebase-adminsdk-hsmqh-b9eb1766fa.json")
        
        ## Object id
        import uuid
        self.object_id = f"FirestoreDB_{uuid.uuid4()}"
        
        ## FirestorApp 
        self.app = firebase_admin.initialize_app(cred, name=self.object_id)
        
        ## Client object
        self.db = firestore.client(self.app)
        
    def close(self):
        # Try to close the object
        firebase_admin.delete_app(self.app)     
       
    def __del__(self):
        print("Cleaning DB session")
        try:
            firebase_admin.delete_app(self.app)
        except: pass
    
    def insert_asset(self, user_id, bot_id, symbol, bot_name=None,shares=None, price=None, fees=None, date=None):
        db= self.db
        
        ## Uncomment in case u do not want to keep the historic of transactions
        #asset_doc = self.get_assed_id(user_id,bot_id,symbol)
        
        #self.delete_asset(asset_doc)
            
        data = {
            'userId': user_id,
            'botId' : bot_id,
            'symbol': symbol
               }
        
        if shares is not None:
            data['botName']  = bot_name
        if shares is not None:
            data['shares']   = shares
        if price is not None:
            data['price']    = price
        if fees is not None:
            data['fees']     = fees
        if date is not None:
            data['date']     = date
            
        doc_ref = db.collection('asset').add(data)

    def delete_asset(self, document_id):
        db = self.db
        doc_ref = db.collection('asset').document(document_id)
    
        if doc_ref.get().exists:
            doc_ref.delete()

    
    def get_assed_id(self, user_id, bot_id, symbol):
        query = self.db.collection('asset').where(field_path = 'userId',
                                                  op_string  = '==', 
                                                  value      = user_id).where(
                                                  field_path = 'botId',
                                                  op_string  = '==', 
                                                  value      =  bot_id).where(
                                                  field_path = 'symbol',
                                                  op_string  = '==',
                                                  value      = symbol)
        
        results = query.stream()   
        
        for doc in results:       
            return doc.id
        
        return None

          

    def create_bot(self, botName, user_id, symbol, end_date, investment,t_limit=None,max_spend=None):
        db = self.db
        data = {
            'botName'   : botName,
            'userId'    : user_id,
            'symbol'    : symbol,
            'endDate'   : end_date,
            'investment': investment,
            'tLimit'    : t_limit,
            'maxSpend'  : max_spend
        }

        doc_ref = db.collection('bot2').add(data)
        
    def delete_bot(self, document_id):
        db = self.db
        doc_ref = db.collection('bot2').document(document_id)
    
        if doc_ref.get().exists:
            doc_ref.delete()
            
    def update_bot(self, document_id, end_date=None, investment=None,acc_value=None):
        db = self.db
        doc_ref = db.collection('bot2').document(document_id) 

        update_data = {}

        if end_date is not None:
            update_data['endDate'] = end_date
            
        if investment is not None:
            update_data['investment'] = investment
            
        if acc_value is not None:
            update_data['accValue'] = acc_value
            
        if update_data:   
            doc_ref.update(update_data) 
                
    def list_collection(self, collection):
        db = self.db

        docs = db.collection(collection).stream()

        docs_list = []

        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['document_id'] = doc.id   
            docs_list.append(doc_data)

        return pd.DataFrame(docs_list)

    def list_active_bots(self, start_date, end_date):
        db = self.db

        query = db.collection('bot2').where('endDate', '>=', end_date)
        docs = query.stream()

        docs_list = []

        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['document_id'] = doc.id 
            docs_list.append(doc_data)

        df = pd.DataFrame(docs_list)

        return df 

        
# Local DB

class SQLiteDB:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
    
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            uid TEXT PRIMARY KEY,
            lastName TEXT,
            metadata TEXT,
            firstName TEXT,
            displayName TEXT,
            contact TEXT,
            address TEXT,
            email TEXT,
            plan TEXT
        )''')
        
     
        cursor.execute('''CREATE TABLE IF NOT EXISTS bots (
            botId TEXT,
            userId TEXT,
            symbol TEXT,
            endDate TEXT,
            investment REAL
        )''')

     
        cursor.execute('''CREATE TABLE IF NOT EXISTS assets (
            symbol TEXT,
            botId TEXT ,
            userId TEXT,
            amount REAL,
            price REAL,
            fees REAL,
            date TEXT,
            PRIMARY KEY (symbol, botId)
        )''')
 
        cursor.execute('''CREATE TABLE IF NOT EXISTS balance (
            userId TEXT PRIMARY KEY,
            currentBalance REAL
        )''')

        self.conn.commit()

   
    def add_user(self, uid, lastName, metadata, firstName, displayName, contact, address, email, plan):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO users (uid, lastName, metadata, firstName, displayName, contact, address, email, plan)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (uid, lastName, metadata, firstName, displayName, contact, address, email, plan))
        self.conn.commit()

    def get_user(self, uid):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM users WHERE uid = ?''', (uid,))
        return cursor.fetchone()

    def update_user(self, uid, lastName=None, metadata=None, firstName=None, displayName=None, contact=None, address=None, email=None, plan=None):
        cursor = self.conn.cursor()
        fields = [f"{field} = ?" for field in ['lastName', 'metadata', 'firstName', 'displayName', 'contact', 'address', 'email', 'plan'] if locals()[field] is not None]
        values = [locals()[field] for field in ['lastName', 'metadata', 'firstName', 'displayName', 'contact', 'address', 'email', 'plan'] if locals()[field] is not None]
        cursor.execute(f'''UPDATE users SET {', '.join(fields)} WHERE uid = ?''', values + [uid])
        self.conn.commit()

    def delete_user(self, uid):
        cursor = self.conn.cursor()
        cursor.execute('''DELETE FROM users WHERE uid = ?''', (uid,))
        self.conn.commit()

  
    def add_bot(self, botId, userId, symbol, endDate, investment):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO bots (botId, userId, symbol, endDate, investment)
                          VALUES (?, ?, ?, ?, ?)''', (botId, userId, symbol, endDate, investment))
        self.conn.commit()

    def get_bot(self, botId):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM bots WHERE botId = ?''', (botId,))
        return cursor.fetchone()

    def update_bot(self, botId, userId=None, symbol=None, endDate=None, investment=None):
        # Construir a parte SET da instrução SQL baseada nos argumentos fornecidos
        fields_to_update = []
        parameters = []

        if userId is not None:
            fields_to_update.append("userId = ?")
            parameters.append(userId)

        if symbol is not None:
            fields_to_update.append("symbol = ?")
            parameters.append(symbol)

        if endDate is not None:
            fields_to_update.append("endDate = ?")
            parameters.append(endDate)

        if investment is not None:
            fields_to_update.append("investment = ?")
            parameters.append(investment)

        # Certificar-se de que pelo menos um campo será atualizado
        if not fields_to_update:
            raise ValueError("No fields to update were provided.")

        set_clause = ", ".join(fields_to_update)
        sql = f"UPDATE bots SET {set_clause} WHERE botId = ?"
        parameters.append(botId)

        # Executar a instrução de atualização
        cursor = self.conn.cursor()
        cursor.execute(sql, parameters)
        self.conn.commit()

    def delete_bot(self, botId):
        cursor = self.conn.cursor()
        cursor.execute('''DELETE FROM bots WHERE botId = ?''', (botId,))
        self.conn.commit()


    def add_asset(self,symbol, botId, userId, amount, price,fees, date):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO assets (symbol, botId, userId, amount, price, date)
                          VALUES (?, ?,?, ?, ?, ?,?)''', (symbol,botId, 
                                                          userId,
                                                          amount,
                                                          price,
                                                          fees,
                                                          date))
        self.conn.commit()

    def get_asset(self, symbol):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM assets WHERE symbol = ?''', (symbol,))
        return cursor.fetchone()

    def update_asset(self, symbol, botId, userId=None, amount=None, price=None, fees=None, date=None):
 
        fields_to_update = []
        parameters = []

        if userId is not None:
            fields_to_update.append("userId = ?")
            parameters.append(userId)

        if amount is not None:
            fields_to_update.append("amount = ?")
            parameters.append(amount)

        if price is not None:
            fields_to_update.append("price = ?")
            parameters.append(price)
        
        if price is not None:
            fields_to_update.append("fees = ?")
            parameters.append(price)     

        if date is not None:
            fields_to_update.append("date = ?")
            parameters.append(date)

      
        if not fields_to_update:
            raise ValueError("No fields to update were provided.")

        set_clause = ", ".join(fields_to_update)
        sql = f"UPDATE assets SET {set_clause} WHERE symbol = ? AND botId = ?"
        
        parameters += [symbol, botId]   
        cursor = self.conn.cursor()
        cursor.execute(sql, parameters)
        self.conn.commit()

    def delete_asset(self, symbol):
        cursor = self.conn.cursor()
        cursor.execute('''DELETE FROM assets WHERE symbol = ?''', (symbol,))
        self.conn.commit()


    def list_users(self):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM users''')
        cols = [column[0] for column in cursor.description]
        users = cursor.fetchall()
        return pd.DataFrame(users, columns=cols)


    def list_bots(self):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM bots ''')
        cols = [column[0] for column in cursor.description]
        bots = cursor.fetchall()
        return pd.DataFrame(bots, columns=cols)

    def list_active_bots(self, until_date):
        sql = "SELECT * FROM bots WHERE endDate >= ?"
        df = pd.read_sql_query(sql, self.conn, params=(until_date,))
        return df
        
    def list_assets(self):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM assets''')
        cols = [column[0] for column in cursor.description]
        assets = cursor.fetchall()
        return pd.DataFrame(assets, columns=cols)

    def list_UserAssets(self,userId):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM assets where userId= ?''',(userId,))
        cols = [column[0] for column in cursor.description]
        assets = cursor.fetchall()
        return pd.DataFrame(assets, columns=cols)
  
    def add_balance(self, userId, currentBalance):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO balance (userId, currentBalance)
                          VALUES (?, ?)''', (userId, currentBalance))
        self.conn.commit()

    def get_balance(self, userId):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT * FROM balance WHERE userId = ?''', (userId,))
        return cursor.fetchone()

    def update_balance(self, userId, currentBalance):
        cursor = self.conn.cursor()
        cursor.execute('''UPDATE balance SET currentBalance = ? WHERE userId = ?''', (currentBalance, userId))
        self.conn.commit()

    def delete_balance(self, userId):
        cursor = self.conn.cursor()
        cursor.execute('''DELETE FROM balance WHERE userId = ?''', (userId,))
        self.conn.commit()
        
    def close(self):
        self.conn.close()