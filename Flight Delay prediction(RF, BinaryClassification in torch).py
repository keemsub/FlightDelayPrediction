#!/usr/bin/env python
# coding: utf-8

# ## **RandomForestClassifier**

# In[ ]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
Jan_2019 = '/content/drive/MyDrive/Flight Delay prediction/Jan_2019_ontime.csv'
Jan_2020 = '/content/drive/MyDrive/Flight Delay prediction/Jan_2020_ontime.csv'


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


# In[ ]:


#data load
df1 = pd.read_csv(Jan_2019)
df2 = pd.read_csv(Jan_2020)


# In[ ]:


#carriers = 항공사, ARR_DEL15 = target
# we will choose 'ARR_DEL15' as the binary target label (0= on time, 1= late)
df1.info()


# 'DAY_OF_MONTH': Day of the month.
# 
# 'DAY_OF_WEEK': Day of the week.
# 
# 'OP_UNIQUE_CARRIER': Unique transport code.
# 
# 'OP_CARRIER_AIRLINE_ID': Unique aviation operator code.
# 
# 'OP_CARRIER': IATA code of the operator.
# 
# 'TAIL_NUM': Tail number.
# 
# 'OP_CARRIER_FL_NUM': Flight number.
# 
# 'ORIGIN_AIRPORT_ID': Origin airport ID.
# 
# 'ORIGIN_AIRPORT_SEQ_ID': Origin airport ID - SEQ.
# 
# 'ORIGIN': Airport of Origin.
# 
# 'DEST_AIRPORT_ID': ID of the destination airport.
# 
# 'DEST_AIRPORT_SEQ_ID': Destination airport ID - SEQ.
# 
# 'DEST': Destination airport.
# 
# 'DEP_TIME': Flight departure time.
# 
# 'DEP_DEL15': Departure delay indicator
# 
# 'DEP_TIME_BLK': block of time (hour) where the match has been postponed.
# 
# 'ARR_TIME': Flight arrival time.
# 
# 'ARR_DEL15': Arrival delay indicator.
# 
# 'CANCELLED': Flight cancellation indicator.
# 
# 'DIVERTED': Indicator if the flight has been diverted.
# 
# 'DISTANCE': Distance between airports.

# In[ ]:


# 'DAY_OF_WEEK': day  / starting from monday. Will be set to 0 if week day and 1 if weekend.
# 'OP_UNIQUE_CARRIER': carrier identifier.
# 'DEP_TIME_BLK': 24h time chunks.
# 'ORIGIN': departure airport identifier.
# 'DEST': destination airport identfier.
# 'DISTANCE': flight length.


# In[ ]:


pd.set_option('display.max_columns', None)
df1.head(3)


# In[ ]:


df1.tail(5)


# In[ ]:


####EDA####


# In[ ]:


set(df1.columns) == set(df2.columns)


# In[ ]:


df = pd.concat([df1,df2])


# In[ ]:


df.head(5)


# In[ ]:


df.isna().sum()


# In[ ]:


df.drop(['Unnamed: 21'], axis=1, inplace = True)


# In[ ]:


def timeToBlock(t):
    block="Nan"
    if(t> 0 and t< 600):
      block= "00010559"
    if(t>559 and t< 700):
      block= "06000659"
    if(t>659 and t< 800):
      block= "07000759"
    if(t>759 and t< 900):
      block= "08000859"
    if(t>859 and t< 1000):
      block= "09000959"
    if(t>959 and t< 1100):
      block= "10001059"
    if(t>1059 and t< 1200):
      block= "11001159"
    if(t>1159 and t< 1300):
      block= "12001259"
    if(t>1259 and t< 1400):
      block= "13001359"
    if(t>1359 and t< 1500):
      block= "14001459"
    if(t>1459 and t< 1600):
      block= "15001559"
    if(t>1559 and t< 1700):
      block= "16001659"
    if(t>1659 and t< 1800):
      block= "17001759"
    if(t>1759 and t< 1900):
      block= "18001859"
    if(t>1859 and t< 2000):
      block= "19001959"
    if(t>1959 and t< 2100):
      block= "20002059"
    if(t>2059 and t< 2200):
      block= "21002159"
    if(t>2159 and t< 2300):
      block= "22002259"
    if(t>2259 and t< 2400):
      block= "23002359"
    return block


# In[ ]:


df['ARR_TIME_BLK']=df['ARR_TIME'].apply(timeToBlock)
df.head()


# In[ ]:


tms_list = list(pd.unique(df['ARR_TIME_BLK']))
df['ARR_TIME_BLK'] = df['ARR_TIME_BLK'].apply(lambda x : tms_list.index(x) + 1)


# In[ ]:


tms_list2 = list(pd.unique(df['DEP_TIME_BLK']))
df['DEP_TIME_BLK'] = df['DEP_TIME_BLK'].apply(lambda x : tms_list2.index(x) + 1)


# In[ ]:


tms_list


# In[ ]:


df.head(5)


# In[ ]:


delayArrPerTimeSlot= df[['ARR_TIME_BLK', 'ARR_DEL15']].groupby(['ARR_TIME_BLK']).mean()
delayArrPerTimeSlot


# In[ ]:


# drop_obj = lambda x: float(x[0:])

# df.ARR_TIME_BLK = df.ARR_TIME_BLK.apply(drop_obj)
# df.head(3)


# In[ ]:


plt.figure(figsize = (15, 15))
sns.heatmap(df.corr(),annot = True, cmap = 'coolwarm')
plt.show()


# In[ ]:


# Delay prediction¶
# Create a dataframe of possible predictors I will remove a list of variables:

# Variables to be predicted; ARR_DEL15 and CANCELLED
# OP_UNIQUE_CARRIER --> Same as OP_CARRIER_AIRLINE_ID
# OP_CARRIER --> also used for carrier identification but not unique so less usefull as OP_CARRIER_AIRLINE_ID
# TAIL_NUM, OP_CARRIER_FL_NUM --> Doesn't make sense to use the flight number as a predictor
# DEP_Time_BLK --> use DEP_TIME
# DEST --> Seems to consist of same information as other included variables
# DEST_AIRPORT_SEQ_ID and ORIGIN_AIRPORT_SEQ_ID --> use constant airport ids for the moment. Could be interesting if there would be trends in airport performance


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.isnull().any()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop(['ARR_DEL15','OP_UNIQUE_CARRIER','CANCELLED','DIVERTED','TAIL_NUM','OP_CARRIER_FL_NUM'
           ,'DEST_AIRPORT_SEQ_ID', 'ORIGIN', 'OP_CARRIER',
          'DEST','ORIGIN_AIRPORT_SEQ_ID'], axis=1)
y = df['ARR_DEL15']
df.dropna(inplace=True)


# In[ ]:


X.info()


# In[ ]:


# X = X.select_dtypes(exclude=['object'])
# test data의 dalay 비율을 살펴보자!(얼마정도의 분포비율로 옳고 그름을 가려내는지 나타내는 분포가 정규분포면 직관적으로 모델의 수치가 정확하기 때문)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


y_test


# In[ ]:


y_test.count()


# In[ ]:


sns.distplot(y_test)


# In[ ]:


y_test = y_test.to_frame(name='ARR_DEL15')


# In[ ]:


sns.countplot(x = "ARR_DEL15", data = y_test)
plt.title("Delay 분포")
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf= RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,y_train)


# In[ ]:


clf.score(X_test, y_test)


# In[ ]:





# ## **BinaryClassification in torch**

# In[ ]:


import pandas as pd
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
Jan_2019 = '/content/drive/MyDrive/Flight Delay prediction/Jan_2019_ontime.csv'
Jan_2020 = '/content/drive/MyDrive/Flight Delay prediction/Jan_2020_ontime.csv'


# In[ ]:


#data load
df1 = pd.read_csv(Jan_2019)
df2 = pd.read_csv(Jan_2020)


# In[ ]:


df = pd.concat([df1,df2])


# In[ ]:


df.isna().sum()


# In[ ]:


df.drop(['Unnamed: 21'], axis=1, inplace = True)


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.isna().any()


# In[ ]:


tms_list = list(pd.unique(df['DEP_TIME_BLK']))
df['DEP_TIME_BLK'] = df['DEP_TIME_BLK'].apply(lambda x : tms_list.index(x) + 1)


# In[ ]:


import seaborn as sns
sns.countplot(x = 'ARR_DEL15', data=df)


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


df['ARR_DEL15'] = df['ARR_DEL15'].astype('category')
encode_map = {
    'Delay': 1,
    'Not Delay': 0
}
df['ARR_DEL15'].replace(encode_map, inplace=True)


# In[ ]:


X = df.drop(['ARR_DEL15','OP_UNIQUE_CARRIER','CANCELLED','DIVERTED','TAIL_NUM','OP_CARRIER_FL_NUM'
           ,'DEST_AIRPORT_SEQ_ID', 'ORIGIN', 'OP_CARRIER',
          'DEST','ORIGIN_AIRPORT_SEQ_ID'], axis=1)
y = df['ARR_DEL15']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)


# In[ ]:


y_train = y_train.to_frame(name='ARR_DEL15')
y_test = y_test.to_frame(name='ARR_DEL15')


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# In[ ]:


## train data
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = TrainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train.values))
## test data    
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = TestData(torch.FloatTensor(X_test))


# In[ ]:


train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)


# In[ ]:


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(10, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        #64x10 and 12x64
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


model = BinaryClassification()
model.to(device)
print(model)


# In[ ]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[ ]:


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


# In[ ]:





# In[ ]:


model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        acc = binary_acc(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')


# In[ ]:


y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


# In[ ]:


confusion_matrix(y_test, y_pred_list)


# In[ ]:


print(classification_report(y_test, y_pred_list))


# In[ ]:





# In[ ]:




