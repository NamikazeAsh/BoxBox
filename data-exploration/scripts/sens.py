import pandas as pd
from pathlib import Path
O=str(Path(__file__).resolve().parent.parent/'data')+'/'
L=pd.read_parquet(O+'laps.parquet'); P=pd.read_parquet(O+'practice.parquet')
for d in (L,P):
    d['lt']=pd.to_timedelta(d['LapTime']).dt.total_seconds(); d['t']=pd.to_timedelta(d['Time']).dt.total_seconds()
    d['ts']=d['TrackStatus'].astype(str)
L=L.sort_values(['Round','LapNumber','t']); L['gap']=L.groupby(['Round','LapNumber'])['t'].diff()
base=L['lt'].notna()&(L['LapNumber']>1)&L['PitInTime'].isna()&L['PitOutTime'].isna()&(L['ts']=='1')&L['Compound'].isin(['SOFT','MEDIUM','HARD'])
print('base',base.sum())
for g in [0.5,1.0,1.5,2.0]:
    print('gap',g,'kept',(base&~(L['gap']<g)).sum(), round((base&~(L['gap']<g)).mean()*100,1))
c=base&~(L['gap']<1.0)
print(L[c]['Compound'].value_counts().to_dict())
pr=c.groupby(L['Event']).mean().sort_values()*100
print('per race retention min/max', pr.round(1).head(4).to_dict(), pr.round(1).tail(3).to_dict())
P=P.sort_values(['Round','Driver','LapNumber'])
P['ok']=P['lt'].notna()&P['PitInTime'].isna()&P['PitOutTime'].isna()&P['Compound'].isin(['SOFT','MEDIUM','HARD'])
P['ok']&=P['lt']<=1.07*P.groupby(['Round','Driver','Stint'])['lt'].transform('median')
P['run']=(~P['ok']|(P.groupby(['Round','Driver'])['Stint'].diff()!=0)).cumsum()
r=P[P['ok']].groupby(['Round','Driver','run']).size()
tot=P.groupby(['Round','Driver']).ngroups
for n in [5,6,8]:
    k=r[r>=n].reset_index().groupby(['Round','Driver']).ngroups
    print('longrun>=',n,'driver weekends',k,'of',tot,round(k/tot*100,1))
