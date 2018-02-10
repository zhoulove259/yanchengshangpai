import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

train = pd.read_table( './data/train_20171215.txt')
test_A = pd.read_table('./data/test_A_20171225.txt')
sample_A = pd.read_table('./data/sample_A_20171225.txt',header=None)
nowtime=datetime.date(2015,3,4)
# print(starttime+datetime.timedelta(days=32))
timelist=[]
l=np.shape(train)[0]-1
i=0
while i<l:
    d=[nowtime,train.loc[i,'date']]
    timelist.append(d)
    while i<l and train.loc[i,'day_of_week']<=train.loc[i+1,'day_of_week']:
        if train.loc[i,'day_of_week']<train.loc[i+1,'day_of_week']:
            nowtime=nowtime + datetime.timedelta(days=1)
            d = [nowtime, train.loc[i+1, 'date']]
            timelist.append(d)
        i = i + 1
    md=train.loc[i,'day_of_week']
    while md<7:
        nowtime = nowtime + datetime.timedelta(days=1)
        d = [nowtime, np.nan]
        timelist.append(d)
        md=md+1
    i=i+1
    if i>=l:
        break
    md=train.loc[i,'day_of_week']
    k=1
    while k<md:
        nowtime = nowtime + datetime.timedelta(days=1)
        d = [nowtime,np.nan]
        timelist.append(d)
        k=k+1
    nowtime = nowtime + datetime.timedelta(days=1)
timelist=timelist[:-3]
# print(timelist)
r_timelist=[]
for t in timelist:
    for br in [1,2,3,4,5]:
        temp=t.copy()
        temp.append(br)
        r_timelist.append(temp)
data=pd.DataFrame(r_timelist)
data.columns=['r_date','date','brand']

train=pd.merge(data,train,how='right',on=['date','brand'])
data=pd.merge(data,train,how='left',on=['r_date','brand','date'])
week=data.groupby('r_date').day_of_week.median()
weeks=pd.DataFrame()
weeks['r_date']=week.index.tolist()
weeks['day_of_week']=week.values
data.drop('day_of_week',axis=1,inplace=True)
data=pd.merge(data,weeks,how='left',on=['r_date'])
for i in range(len(data)):
    if(~(data.loc[i,'day_of_week']>0)):
        if(data.loc[i,'r_date']==data.loc[i-1,'r_date']):
            data.loc[i, 'day_of_week']= data.loc[i-1, 'day_of_week']
        else:
            if(data.loc[i-1,'day_of_week']==7):
                data.loc[i , 'day_of_week'] =1
            else:
                data.loc[i, 'day_of_week'] = data.loc[i-1, 'day_of_week']+1

def dorelax_day_mean(a):
    tmp=a.copy()
    flag=tmp[0]%10
    d=pd.DataFrame(tmp)
    d.columns=['cnt']
    d['is_relax_day']=d['cnt']%10
    d['cnt'] = (d['cnt']/ 10).astype(int)
    d['cnt'].replace(0,np.nan,inplace=True)
    return np.mean(d.loc[d['is_relax_day']==flag,'cnt'])
def roll_day(data,day):
    return data['cnt'].rolling(window=day,min_periods=1).mean()
def roll_dorelax_day(data,day):
    result=pd.rolling_apply(data['dorelax_day_mean'],window=day,func=dorelax_day_mean,min_periods=1)
    # data.loc[data['is_relax_day'] == 0, 'dorelax_day_mean'] = pd.rolling_apply(data.loc[data['is_relax_day'] == 0, 'cnt'], window=day, func=dorelax_day_mean, min_periods=1)
    return result
relax_day=[6,7]
data['is_relax_day']=0
data.loc[data.day_of_week.isin(relax_day),'is_relax_day']=1
data.loc[~data.day_of_week.isin(relax_day),'is_relax_day']=0
data['dorelax_day_mean']=data['cnt'].fillna(0)*10+data['is_relax_day']
for br in [1,2,3,4,5]:
    data.loc[data['brand'] == br, 'weekmean'] = roll_day(data[data['brand'] == br], 7)
    data.loc[data['brand'] == br, 'monthmean'] = roll_day(data[data['brand'] == br], 30)
    data.loc[data['brand'] == br,'dorelax_day_mean'] = roll_dorelax_day(data[data['brand'] == br], 30)
    # data.loc[data['brand'] == br,'brand_mean'] = data.loc[data['brand'] == br,'cnt'].mean()
    # for w in [1, 2, 3, 4, 5, 6, 7]:
    #     data.loc[(data['day_of_week'] == w)&(data['brand'] == br), 'week_day_brand_mean'] = data.loc[(data['day_of_week'] == w)&(data['brand'] == br), 'cnt'].mean()
# for w in [1,2,3,4,5,6,7]:
#     data.loc[data['day_of_week'] == w, 'week_day_mean'] = data.loc[data['day_of_week'] == br, 'cnt'].mean()

# for br in [1,2,3,4,5]
#     data.loc[data['brand'] == br, 'week_relax_mean'] = roll_dorelax_day(data[data['brand'] == br], 7)
#     data.loc[data['brand'] == br, 'month_relax_mean'] = roll_dorelax_day(data[data['brand'] == br], 30)
data['month']=data['r_date'].apply(lambda x:x.month)
data['year']=data['r_date'].apply(lambda x:x.year)


day_of_weeks = pd.get_dummies(data.day_of_week)
day_of_weeks.columns = ['day_of_week' + str(i + 1) for i in range(day_of_weeks.shape[1])]
data = pd.concat([data, day_of_weeks], axis=1)

months = pd.get_dummies(data.month)
months.columns = ['month' + str(i + 1) for i in range(months.shape[1])]
data = pd.concat([data, months], axis=1)

years = pd.get_dummies(data.year)
years.columns = ['year' + str(i + 1) for i in range(years.shape[1])]
data = pd.concat([data, years], axis=1)
brands = pd.get_dummies(data.brand)
brands.columns = ['brand' + str(i + 1) for i in range(brands.shape[1])]
data = pd.concat([data, brands], axis=1)
# data.drop(['month','year','day_of_week','brand'],inplace=True,axis=1)
data['r_date']=pd.to_datetime(data['r_date'])
print(data.info())
data.to_csv('data/data.csv',index=None)
