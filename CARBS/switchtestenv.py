import gym, os, json
import numpy as np
import rtbenv, pickle
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, SGD

import tensorflow as tf

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TrainEpisodeLogger, FileLogger, ModelIntervalCheckpoint

def build_callbacks(pid='0'):
    log_filename = 'interval_log{0}'.format(pid)
    callbacks = [TrainEpisodeLogger()]
    callbacks += [SelFileLogger(log_filename)]
    return callbacks

class SelFileLogger(FileLogger):
    def __init__(self, filepath, interval=None):
        super().__init__(filepath, interval)
    def save_data(self):
        if len(self.data.keys()) == 0:
            return
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            sorted_data[key] = np.array([self.data[key][idx] for idx in sorted_indexes]).tolist()
        with open(self.filepath, 'a+') as f:
            json.dump(sorted_data, f)

pid = input('Enter Process: ')
ver = '0'#input('Enter Version/GPU: ')
os.environ["CUDA_VISIBLE_DEVICES"] = ver
bu = float(input("Enter Test Budget for hour: "))
pricetype = int(input("Enter price type:　"))
env = gym.make('rtbenv-v0')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)
aid = input('Enter aid: ')
if aid == '1458':
    avgreq = 18612
    avgcpm = 67.86
    data = 'ipin'
elif aid == '3386':
    avgreq = 17520
    avgcpm = 72.28
    data = 'ipin'
elif aid == '3427':
    avgreq = 14842
    avgcpm = 78.99
    data = 'ipin'
elif aid == '215':
    avgreq = 1633
    avgcpm = 55.22
    data = 'tenmax'

numloop = int(input("Enter loop number (100w): "))
numloop *= 1000000
alp = float(input("Enter alpha value: "))
#budget, num_req, num_strategies, starttime, endtime, aid = None, gaphour = 1, dataset = 'ipin', simple = False
if data == 'ipin':
    st =  datetime.datetime.strptime('2013-'+'06'+'-'+'06', '%Y-%m-%d')
    et = st + datetime.timedelta(hours=48)  #also need to change Data(), hard code to 6
elif data == 'tenmax':
    st =  datetime.datetime.strptime('2016-'+'10'+'-'+'01', '%Y-%m-%d')
    et = st + datetime.timedelta(hours=72)
    
env.__init__(bu, 10., 5, st, et, aid = aid, simple = False, dataset = data, skip = 0., avgcpm = avgcpm, avgreq = avgreq, alpha = alp, pricetype = pricetype)
#print (env, env.observation_space.high)

#env.dataset.get_data()

#for i in range(10):
#    env.step(0)

epi_step = 200000
nb_actions = int(env.action_space.n)
print(nb_actions)
memory = SequentialMemory(limit=150000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test = 0., nb_steps = numloop)#EpsGreedyQPolicy(0.3)#BoltzmannQPolicy()
#policy = EpsGreedyQPolicy(0.3)
model = Sequential()
model.add(Flatten(input_shape=(1,) +env.observation_space.shape))
model.add(Dense(70))
model.add(Activation('sigmoid'))
model.add(Dense(35))
#model.add(Activation('sigmoid'))
model.add(Dense(15))
#model.add(Dense(15))
#model.add(Activation('sigmoid'))
model.add(Dense(5, activation='linear'))
#dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn = DQNAgent(model=model, nb_actions=nb_actions, gamma = 1., nb_steps_warmup=10000, memory=memory, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn.load_weights('weights/10_0_3600000.h5')
nb_steps=numloop
count = 0#3800000
#epi_step = 200000
#while(count <= nb_steps):
    #callbacks = build_callbacks(pid)
env.dataset.start =  st#datetime.datetime.strptime('2013-'+'06'+'-'+'06', '%Y-%m-%d')
env.dataset.end = et#env.dataset.start + datetime.timedelta(hours=1)
env.realstart = env.dataset.start
env.realend = env.dataset.end



env.train = 1
env.dataset.train = 1
weights_filename = 'weights/{1}_{0}'.format(ver, pid)+'_{step}.h5f'
final_weights_filename = 'weights/final_{1}_{0}'.format(ver, pid)+'.h5f'
log_filename = "log/log{1}_{0}".format(ver, pid)+".json"
callbacks = [ModelIntervalCheckpoint(weights_filename, interval=epi_step)]
callbacks += [FileLogger(log_filename)]#epi_step)]
callbacks += [TrainEpisodeLogger()]

dqn.fit(env, nb_steps = numloop, callbacks = callbacks, verbose = 1)
dqn.save_weights(final_weights_filename, overwrite=True)
#print(memory.observations.data, len(memory.observations) )
#dqn.fit(env, nb_steps = numloop)
#dqn.save_weights('weights/{2}_{0}_{1}.h5'.format(ver, count, pid), overwrite=True)

with open('log/log{1}_{0}.json'.format(ver, pid), 'r') as re:
    logs = json.load(re)
maxrewardind = logs['episode_reward'].index(max(logs['episode_reward']))
select = (logs['nb_steps'][maxrewardind] - logs['nb_episode_steps'][maxrewardind])//epi_step
if (logs['nb_steps'][maxrewardind] - logs['nb_episode_steps'][maxrewardind])//epi_step >= (select+0.5):
    select += 1
if select == 0:
    select += 1
print(select)
dqn.load_weights('weights/{0}_{1}_{2}.h5f'.format(pid, ver, int(select*epi_step)))
'''
if data == 'ipin':
    env.dataset.start = st + datetime.timedelta(days=2)#+datetime.timedelta(hours=8)
    env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*24)
elif data == 'tenmax':
    env.dataset.start = st + datetime.timedelta(days=3)
    env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*24)
env.realstart = env.dataset.start
env.realend = env.dataset.end
env.train = 0
env.dataset.train = 0

dqn.test(env, nb_episodes=1, visualize=False)
#with open('testrecord.pkl', 'wb') as ddd:
#	pickle.dump(env.testrecord, ddd)
with open("log/valh{1}-{0}-{2}.txt".format(ver, pid, select), 'w') as d:
    for i in env.act_record:
        d.write(str(i[0])+"\t"+str(i[1])+"\t"+str(i[2])+"\t"+str(i[4])+"\t"+str(i[5])+"\t"+str(i[6])+"\t"+str(i[7])+"\t"+str(i[8])+'\t'+str(i[9])+'\t'+str(i[12])+'\n')
#count += epi_step
#print("fit ", count, " steps")
env.record_result(pid)
print("highest Val.,", env.dataset.start, env.dataset.end)
print("Total step, ", dqn.step)
print("clicks: ", env.click, "cost: ", env.cost, "Imp: ", env.imp)
'''
if data == 'ipin':
    env.dataset.start = st + datetime.timedelta(days=3)
    env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*96)
elif data == 'tenmax':
    env.dataset.start = st + datetime.timedelta(days=4)
    env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*72)
env.realstart = env.dataset.start
env.realend = env.dataset.end
env.train = 0
env.dataset.train = 0
print(env.dataset.start, env.dataset.end)
dqn.test(env, nb_episodes=1, visualize=False)
with open('testrecord.pkl', 'wb') as ddd:
    pickle.dump(env.testrecord, ddd)

with open("log/testh{1}-{0}-{2}.txt".format(ver, pid, select), 'w') as d:
    for i in env.act_record:
        d.write(str(i[0])+"\t"+str(i[1])+"\t"+str(i[2])+"\t"+str(i[4])+"\t"+str(i[5])+"\t"+str(i[6])+"\t"+str(i[7])+"\t"+str(i[8])+'\t'+str(i[9])+'\t'+str(i[12])+'\n')
env.record_result(pid)
print("Total step, ", dqn.step)
print("clicks: ", env.click, "cost: ", env.cost, "Imp: ", env.imp)


'''
dqn.load_weights('weights/final_{1}_{0}.h5f'.format(ver, pid))
#env.dataset.start = st + datetime.timedelta(days=2)#+datetime.timedelta(hours=8)
#env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*24)
if data == 'ipin':
    env.dataset.start = st + datetime.timedelta(days=2)#+datetime.timedelta(hours=8)
    env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*24)
elif data == 'tenmax':
    env.dataset.start = st + datetime.timedelta(days=3)
    env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*24)
env.realstart = env.dataset.start
env.realend = env.dataset.end
env.train = 0
env.dataset.train = 0
print("final, ", env.dataset.start, env.dataset.end)
dqn.test(env, nb_episodes=1, visualize=False)
with open("log/valf{1}-{0}.txt".format(ver, pid), 'w') as d:
    for i in env.act_record:
        d.write(str(i[0])+"\t"+str(i[1])+"\t"+str(i[2])+"\t"+str(i[4])+"\t"+str(i[5])+"\t"+str(i[6])+"\t"+str(i[7])+"\t"+str(i[8])+'\t'+str(i[9])+'\t'+str(i[12])+'\n')
#count += epi_step
#print("fit ", count, " steps")
env.record_result(pid)
print("Total step, ", dqn.step)
print("clicks: ", env.click, "cost: ", env.cost, "Imp: ", env.imp)


#env.dataset.start = st + datetime.timedelta(days=3)
#env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*96)
if data == 'ipin':
    env.dataset.start = st + datetime.timedelta(days=3)
    env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*96)
elif data == 'tenmax':
    env.dataset.start = st + datetime.timedelta(days=4)
    env.dataset.end = env.dataset.start + datetime.timedelta(hours=env.dataset.gaphour*72)

env.realstart = env.dataset.start
env.realend = env.dataset.end
env.train = 0
env.dataset.train = 0
print(env.dataset.start, env.dataset.end)
dqn.test(env, nb_episodes=1, visualize=False)

with open("log/testf{1}-{0}.txt".format(ver, pid), 'w') as d:
    for i in env.act_record:
        d.write(str(i[0])+"\t"+str(i[1])+"\t"+str(i[2])+"\t"+str(i[4])+"\t"+str(i[5])+"\t"+str(i[6])+"\t"+str(i[7])+"\t"+str(i[8])+'\t'+str(i[9])+'\t'+str(i[12])+'\n')
env.record_result(pid)
print("Total step, ", dqn.step)
print("clicks: ", env.click, "cost: ", env.cost, "Imp: ", env.imp)'''
