import  pandas as pd
import numpy as np

import mxnet
import mxnet.gluon as G

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

#Get set up
from pandas.io.json import json_normalize


#from tqdm import tqdm_notebook as tqdm
import time
import datetime
#8:27am
import warnings
#warnings.filterwarnings('ignore')

import os
from sqlalchemy import create_engine as __create_engine


def create_engine_for_db(*args, **kwargs):
    username = 'admin'
    password = 'mids2019'
    host = 'capstone-db.chkep37zmim2.us-east-2.rds.amazonaws.com'
    schema = 'market_data'

#def create_engine_for_db(*args, **kwargs):
#    username = os.environ['DB_USER']
#    password = os.environ['DB_PASS']
#    host = os.environ['DB_HOST']
#    schema = os.environ['DB_SCHEMA']
    
    DB_URL = f"mysql://{username}:{password}@{host}/{schema}"
    return __create_engine(DB_URL, *args, **kwargs)


engine = create_engine_for_db()


class Net(G.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            #change the 2 to 1
            self.lstm = G.rnn.LSTM(10,2,dropout=0.2)
            self.out = G.nn.Dense(1)
            
    def forward(self,x):
        
        x = self.lstm(x)
        x = self.out(x)
        
        return x
    
Model = Net()
print(Model)


sql = '''   SELECT distinct slug
     from lstm_control 
     where machine_id = 1
'''
slugs = pd.read_sql(sql, con=engine)



#new
df_lstm_output = pd.DataFrame(columns=['id','date','roi','predicted_roi'])
for index, i in slugs.iterrows():
    engine = create_engine_for_db()
    sql = '''   SELECT *
      FROM daily_tbl
      where id = %(id)s
'''
    
#    df_coin = pd.read_sql(sql, con=engine)
    df_coin = pd.read_sql(sql, con=engine,  params={'id': i['slug']})
#    coin = i
#    df_coin = df[df['id'] == coin].copy()
    df_coin.dropna(inplace=True)
    df_coin.sort_values(by=['date'],inplace=True)
    df_coin['roi'] = df_coin['close'].pct_change()
    df_coin.dropna(inplace=True)
    df_output = df_coin
    df_output['predicted_roi']=0.0
    print(datetime.datetime.now())
    print(i['slug'])
    for i in range(50, len(df_coin)):
        df_train = df_coin.iloc[i-50:i,:]
        test = df_coin.iloc[i,:]
        df_train.reset_index(inplace=True)
        trn_x = df_train[['open', 'high',
           'low', 'close', 'volume', 'market', 'spread', 'close_ratio', 'mom30',
           'mom60', 'mom90', 'mom180', 'market_signal5', 'market_signal20',
           'market_signal30', 'market_signal60']].copy()
        trn_y = df_train['roi'].copy()

        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(trn_x)
        trn_x= scaler.transform(trn_x)
        trn_x = trn_x.reshape(50,16,1)
        trn_x = mxnet.nd.array(trn_x)
        trn_y = mxnet.nd.array(trn_y)
        #print(f'train shape {trn_x.shape,trn_y.shape}')
        device = mxnet.gpu(0) if mxnet.context.num_gpus() > 0 else mxnet.cpu(0)
    #    Model.initialize(mxnet.init.Xavier(),ctx=device)
        Model.initialize(mxnet.init.Xavier(),ctx=device, force_reinit=True)
        #print(device)


        trainer = G.Trainer(
        params=Model.collect_params(),
        optimizer='adam',
        optimizer_params={'learning_rate': 0.001},
    )

        loss_function = G.loss.L2Loss()
        mse = mxnet.metric.MSE()

        from mxnet import autograd
        EPOCHS = 1000
        trn_loss = []
        train_iter = mxnet.io.NDArrayIter(trn_x, trn_y, 50, shuffle=False)
        for epoch in range(EPOCHS):
            for trn_batch in train_iter:

                x = trn_batch.data[0].as_in_context(device)
                y = trn_batch.label[0].as_in_context(device)

                with autograd.record():
                    y_pred = Model(x)
                    #print(Model)
                    loss = loss_function(y_pred, y)
                    #print(f'original y {y}')
                    #print(f"y_pred {y_pred}")
                    #print(f"loss{loss}")

                #backprop
                loss.backward()

                #Optimize!
                trainer.step(batch_size=trn_x.shape[0])
                #break

            train_iter.reset()

            # Calculate train metrics

            predictions = Model(trn_x.as_in_context(device))
            #print(f"train_y {trn_y}, prediction {predictions}")
            mse.update(trn_y, predictions)
            trn_loss.append(mse.get()[1])
            mse.reset()

            train_iter.reset()

        #    print("epoch: {} | trn_loss: {:.8f}".format(epoch+1,
        #                                                trn_loss[-1]))
            test2 = test[['open', 'high',
               'low', 'close', 'volume', 'market', 'spread', 'close_ratio', 'mom30',
               'mom60', 'mom90', 'mom180', 'market_signal5', 'market_signal20',
               'market_signal30', 'market_signal60']].values.reshape(1,16,1)

            test2 = mxnet.nd.array(test2)
            #print(test2.shape)
            Model(test2.as_in_context(device))

        out = Model(test2.as_in_context(device)).asnumpy()
        df_lstm_output=df_lstm_output.append({'id':test['id'], 'date':test['date'],'roi':test['roi'],'predicted_roi':out.item()}, ignore_index=True)

    print('done')
    print(i)
    print(datetime.datetime.now())
    df_lstm_output.to_csv('lstm14.csv', index=False)
    engine = create_engine_for_db()
    df_lstm_output.to_sql('lstm_tbl', schema='market_data', con=engine,   if_exists='append',index=False)
    df_lstm_output = pd.DataFrame(columns=['id','date','roi','predicted_roi'])

