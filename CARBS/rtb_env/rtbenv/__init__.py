import logging
from gym.envs.registration import register
import datetime

a = 0 

if a == 0:
    st =  datetime.datetime.strptime('2013-'+'06'+'-'+'06', '%Y-%m-%d')
else:
    st =  datetime.datetime.strptime('2016-'+'10'+'-'+'01', '%Y-%m-%d')
et = st + datetime.timedelta(hours=1)

if a == 0:
    register( id='rtbenv-v0', entry_point='rtbenv.envs:RTBEnv', kwargs={'budget': 1000, 'num_req': 1000, 'num_strategies': 5, 'dataset':'ipin', 'starttime':st, 'endtime':et, 'aid':'1458', 'simple':False} )
else:
    register( id='rtbenv-v0', entry_point='rtbenv.envs:RTBEnv', kwargs={'budget': 1000, 'num_req': 1000, 'num_strategies': 5, 'dataset':'tenmax', 'starttime':st, 'endtime':et, 'aid':'215', 'simple':False} )
