#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym, pickle, sys
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
from pymongo import MongoClient
import datetime
import random
#sys.path.append('/home/wyshih/rltemp/wrmodel/ipin')
#sys.path.append('/home/wyshih/rltemp/wrmodel/tmax')
sys.stdout.flush()
class Data():

    def __init__(self, starttime, endtime, train = False, aid = None, gaphour = 1, dataset = 'ipin', simple = False, skip = 0.):
        self.client = MongoClient('Enter your own data source')
        self.train = train
        self.skip = skip
        self.start = starttime
        self.end = endtime
        self.current = starttime
        self.aid = aid
        self.gaphour = gaphour
        self.dataset = dataset
        self.simple = simple
        self.cursor = None
        self.get_data()
        print("Init Dataset")

    def get_data(self):
        dataset = self.dataset
        simple = self.simple
        gaphour = self.gaphour
        aid = self.aid
        
        if dataset == 'ipin' and simple:
            db = self.client.simplertb.ipinyou
        elif dataset == 'ipin' and simple == False:
            db = self.client.iPinYou_allinone.ipinyou
        elif dataset == 'tenmax' and simple:
            db = self.client.simplertb.doubleclick
        elif dataset == 'tenmax' and simple == False:
            db = self.client.DoubleClick.doubleclick
        time1 = self.start
        time2 = time1 + datetime.timedelta(hours=gaphour)
        #print(time1, time2, dataset, simple)
        if dataset == 'ipin':
            if simple:
                if aid!=None:
                    self.cursor = db.find({'timestamp':{"$gte": time1, '$lt': time2}, "paying_price": {"$gt":0.}, 'advertiser_id':aid})
                else:
                    self.cursor = db.find({'timestamp':{"$gte": time1, '$lt': time2}, "paying_price": {"$gt":0.}})
            else:
                if aid != None:
                    self.cursor = db.find({'timestamp':{"$gte": time1, '$lt': time2}, "bidding_price": {"$gt": 0.}, "paying_price": {"$gt":0.}, 'advertiser_id':aid, "lg_ctr": {"$exists":True}})
                else:
                    self.cursor = db.find({'timestamp':{"$gte": time1, '$lt': time2}, "bidding_price": {"$gt": 0.}, "paying_price": {"$gt":0.}, "lg_ctr": {"$exists":True}})

        elif dataset == 'tenmax':
            if simple:
                if aid!=None:
                    self.cursor = db.find({'dateTime':{"$gte": time1, '$lt': time2}, "winEvent.price": {"$gt":0.}, 'bidResponse.advertiserId':aid})
                else:
                    self.cursor = db.find({'dateTime':{"$gte": time1, '$lt': time2}, "winEvent.price": {"$gt":0.}})

            else:
                if aid!=None:
                    self.cursor = db.find({'dateTime':{"$gte": time1, '$lt': time2}, "bidResponse.price": {"$gt": 0.}, "winEvent.price": {"$gt":0.}, 'bidResponse.advertiserId':aid})
                else:
                    self.cursor = db.find({'dateTime':{"$gte": time1, '$lt': time2}, "bidResponse.price": {"$gt": 0.}, "winEvent.price": {"$gt":0.}})

        #print("Dataset, ", self.dataset, "aid, ", self.aid, "records, ", self.cursor.count())

    def next_record(self):
        count = 0
        try:
            inst = self.cursor.next()
            
            count = 1
            if self.train:
                if self.dataset == 'ipin':
                    while random.random()<= self.skip and len(inst.keys()) != 0 and inst['click'] == False:
                        #count += 1
                        inst = self.cursor.next()
                        count += 1
                    
            return inst, count
        except StopIteration:
            return {}, count 
    def rest_time(self, instance):
        if len(instance.keys()) != 0:
            end = self.start + datetime.timedelta(hours=self.gaphour)
            if self.dataset == 'ipin':
                rest = end - instance['timestamp']#current  
            elif self.dataset == 'tenmax':
                rest = end - instance['dateTime']
            return rest.total_seconds()/datetime.timedelta(hours=self.gaphour).total_seconds()
        else:
            return 0
            
class RTBEnv(gym.Env):

    #metadata = {'render.modes': ['human']}

    def __init__(self, budget, num_req, num_strategies, starttime, endtime, aid = None, gaphour = 1, dataset = 'ipin', simple = False, skip = 0., avgcpm = 78.99, avgreq = 14842, alpha = 0.5, retest = False, pricetype = 2):
        if dataset == 'ipin':
            print('ipin')
            sys.path.append('/home/wyshih/rltemp/wrmodel/ipin')
        else:
            print('tenmax')
            while ('/home/wyshih/rltemp/wrmodel/ipin' in sys.path):
                sys.path.remove('/home/wyshih/rltemp/wrmodel/ipin')
            sys.path.append('/home/wyshih/rltemp/wrmodel/tmax')
       
        self.total_budget = budget*(((endtime-starttime).total_seconds()/3600)/gaphour)
        self.test_budget = budget
        self.hourly_budget = self.total_budget/(((endtime-starttime).total_seconds()/3600)/gaphour)
        self.total_num_req = num_req
        self.budget = 0.#budget
        self.current_budget = self.total_budget 
        self.remain_time = 1.
        self.realstart = starttime
        self.realend = endtime
        self.dataset = Data(starttime, endtime, True, aid, gaphour, dataset, simple, skip)
        self.remain_req = self.dataset.cursor.count()
        self.total_req = self.dataset.cursor.count()
        self.end_episode = False
        self.episode = 0
        self.click = 0
        self.imp = 0
        self.cost = 0.
        self.bid = 0.
        #self.skip = 0.1
        self.instance, count = self.dataset.next_record()
        self.remain_req -= count
        #if simple:
        #    self.ctr_flag = self.instance['ctr']
        #else:
        #    self.ctr_flag = self.instance['lg_ctr']
        self.feeddata = False
        self.action_space = spaces.Discrete(num_strategies)
        self.act_record = []
        self.train = 1
        self.avgcpm = avgcpm
        self.avgreq = avgreq
        self.alpha = alpha
        self.totalreward = 0.
        self.pricetype = pricetype
        self.turnon = False
        self.count = 0.
        self.specount = 0.
        self.speclu = -1
        self.discount = 0.
        self.wpcount = 0.
        self.moniwp = 0.
        self.testrecord = []
        self.retest = retest
        #low = np.array([130., 0., 0., 0])
        #low = np.array([130., 0., 0., 0, 0])
        low = np.array([1., 0, 0, 0, 0, 0, 0, 0, 0]+[0]*24)
        #high = np.array([0., self.remain_time, self.remain_req, 1.])
        #high = np.array([0., self.remain_time, 1., 23, 7])
        high = np.array([0., 1, 1, 1, 1, 1, 1, 1, 1]+[1]*24)
        if dataset == 'ipin':
            with open('wrmodel/ipin/{0}.model'.format(aid), 'rb') as data:
                self.model, self.bidf, self.f = pickle.load(data)
        elif dataset == 'tenmax':
            with open('wrmodel/tmax/{0}.model'.format(aid), 'rb') as data:
                self.model, self.bidf, self.f = pickle.load(data)

        self.ctr_flag = 0#self.model.ctrmodel.predict_proba(self.model.data_prepare(self.instance, pow(2, 20), ctrtrain = True))[0][1]
        self.wrtarget = self.cal_wr_b(self.avgreq, self.avgcpm)#(13300, 178.)#22000, 150.)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
    def _adjust(self, act):
        convert = [0., -0.025, -0.05, -0.075, -0.1]#, -0.2, -0.25]#self._find_action_table(self.wrtarget[self.gen_s(self.ctr_flag)])#[-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
        return convert[int(act)]
    def _find_action_table(self, m):
        left = int(min(m-0, 0.1)//0.025)
        right = int(min(1-m, 0.1)//0.025)
        if left > right:
            left = 4 - right
        elif left < right:
            right = 4 - left
        acts = [0.025*i for i in range(1, right+1)]
        acts.extend([-0.025*i for i in range(1,left+1)])
        acts.append(0.)
        return acts         
    def _action(self, act, instance):

        '''if act == 0:
            if instance['ctr'] > 0.01:
                return 150
            return 1.
            #lamb = 5.2*pow(10, -7)
            #if self.dataset.simple:
            #    return pow(80*instance['ctr']/lamb+pow(80, 2), 0.5)-80
            #else:
            #    return pow(80*instance['lg_ctr']/lamb+pow(80, 2), 0.5)-80
        elif act == 1:
            lamb = 5.2*pow(10, -6)
            if self.dataset.simple:
                return pow(50*instance['ctr']/lamb+pow(50, 2), 0.5)-50    
            else:
                return pow(50*instance['lg_ctr']/lamb+pow(50, 2), 0.5)-50
        else:
            return 1.
        '''
        #wr = (float(act)*1.)/10.
        wr = min(max(act, 0.), 1.)
        if wr == 0:
            wr = 0.00001
        if wr == 1:
            wr = 0.99999        
        data = self.model.data_prepare(self.instance, self.model.feature_size, False)
        price = self.model.recommend_price_wr(data, self.model.feature_size, self.bidf.coef_[0], self.bidf.intercept_[0], wr)
        return price

        '''if act == 0:
            return 150.
        
        elif act == 1:
            return 100.
        elif act == 2:
            return 80
        elif act == 3:
            return 20
        else:
            return 1.
        '''
    def step(self, act):
        instance = self.instance#dataset.next_record()
        #print self.remain_req, act
        '''while(random.random()<self.skip and self.remain_req > 0):
            instance = self.dataset.next_record()
            self.remain_req -= 1
            if len(instance.keys()) == 0:
                print self.remain_req, 'loop'
                break
        '''
        '''if len(instance.keys()) == 0 and (self.dataset.start < self.dataset.end-datetime.timedelta(hours=self.dataset.gaphour)):
            print("End Here", self.dataset.start, self.dataset.end)
            self.feeddata = True
            self.dataset.start += datetime.timedelta(hours=self.dataset.gaphour)

        if self.feeddata:
            self.dataset.get_data()
            self.remain_req = self.dataset.cursor.count() 
            instance = self.dataset.next_record()
            print("Here", self.dataset.start, self.dataset.end)
            self.feeddata = False
        '''
        if (len(instance.keys()) == 0 or self.budget < self.avgcpm or self.remain_time <= 0 or self.remain_req == 0) and self.dataset.start >= self.dataset.end-datetime.timedelta(hours=self.dataset.gaphour):
            self.end_episode = True
            print("End Episode! \n", self.dataset.start)
            ob = self._get_state()
            return ob, 0, self.end_episode, {}

        #self.remain_req -= 1
        
        #try:
        reward = 0
        if self.dataset.dataset == 'ipin':
            try:
                instance['ctr_flag'] = self.model.ctrmodel.predict_proba(self.model.data_prepare(instance, pow(2, 20), ctrtrain = True))[0][1]
            except:
                print('req', self.remain_req, self.remain_time, self.dataset.start, self.dataset.end, self.budget, instance, self.instance, len(instance.keys()))
        self.ctr_flag = self.model.ctrmodel.predict_proba(self.model.data_prepare(self.instance, pow(2, 20), ctrtrain = True))[0][1]
        if self.dataset.dataset == 'ipin':
            if instance['ctr_flag'] != self.ctr_flag:
                print('CTR not match')
            if self.dataset.start.hour != instance['timestamp'].hour:
                print('change hour!!')
        #bid = self._action(act, instance)
        adj = self._adjust(act)
        bid = self._action(self.wrtarget[self.gen_s(self.ctr_flag)]+adj, instance)
        if self.dataset.dataset == 'ipin':
            wp = instance['paying_price']
        elif self.dataset.dataset == 'tenmax':
            wp = round(math.exp(math.log(float(instance['winEvent']['price'])/1.2)+5.6)+2)
        #except:
        #    print(instance, self.feeddata,  self.dataset.cursor.count(), self.dataset.start, self.remain_req)
        if self.dataset.dataset == 'ipin':
            try:
                self.dataset.current = instance['timestamp']
            except:
                print (instance, self.feeddata,  self.dataset.cursor.count(), self.dataset.start, self.remain_req, 'timestamp')
        else:
            self.dataset.current = instance['dateTime']    

        self.remain_time = self.dataset.rest_time(self.instance)
        if self.dataset.dataset == 'ipin':
            click = instance['click']
        elif self.dataset.dataset == 'tenmax':
            if instance.get('clickEvent'):
                if instance['clickEvent']['click'] == True:
                    click = True
                else:
                    click = False
            else:
                click = False
        
        #self.ctr_flag = instance['ctr']    
        cwr = float(act)*1./10
        cctr = self.get_reward()
        #score = {0: 1., 1:1., 2:1., 3: 1e-9, 4: 1, 5: 1e-9, 6: 1., 7: 0.1}
        #bid = min(self.budget, bid)
        '''if self.train == False and self.retest and self.turnon:
            self.count += 1.
            if self.wrtarget[self.gen_s(self.ctr_flag)] > 0 and self.wrtarget[self.gen_s(self.ctr_flag)] < 1:
                self.specount += 1.   
                if self.speclu != self.gen_s(self.ctr_flag) and self.speclu != -1:
                    print("Wrong monitor!")
                    return -1
                if self.wpcount > 1000:
                    #monitor = self.specount/self.count
                    #if self.speclu != -1 and self.model.clusters[3][self.speclu] < monitor:
                    monitor = self.moniwp/self.cost
                    truthallocate = 1.*self.avgreq*self.model.clusters[3][self.speclu]*self.model.clusters[4][self.speclu][self.pricetype]*self.wrtarget[self.gen_s(self.ctr_flag)]/self.hourly_budget
                    if self.speclu != -1 and monitor > truthallocate:
                        self.discount = monitor / truthallocate#max(monitor/self.model.clusters[3][self.speclu], self.discount)
                        bid = self._action((self.wrtarget[self.gen_s(self.ctr_flag)]+adj)/self.discount, instance)
                        #print(monitor, truthallocate, self.discount,self.avgreq,self.model.clusters[3][self.speclu],self.model.clusters[4][self.speclu][2],self.wrtarget[self.gen_s(self.ctr_flag)],self.hourly_budget)
        '''
        if bid > self.budget and self.train == False:
            bid = min(bid, self.budget)
        if self.train == False:
            self.testrecord.append([instance['bid_id'], bid,  self.gen_s(self.ctr_flag), self.ctr_flag])
        #reward = (cctr - abs(cwr-cctr)) * (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)])#1 - abs(cwr-cctr)
        '''if cctr <= self.wrtarget[self.gen_s(self.ctr_flag)] and cwr >= cctr and cwr <= self.wrtarget[self.gen_s(self.ctr_flag)]:
            reward = cctr * (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)])
        elif cctr > self.wrtarget[self.gen_s(self.ctr_flag)] and cwr <= cctr and cwr >= self.wrtarget[self.gen_s(self.ctr_flag)]:
            reward = cctr * (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)])
        else:
            reward = (cctr - abs(cwr-self.wrtarget[self.gen_s(self.ctr_flag)])) * (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)])
        '''
        #reward = min(max(self.wrtarget[self.gen_s(self.ctr_flag)]+self._adjust(act), 0.), 1.) * cctr * (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)])/bid
        if self.dataset.dataset == 'ipin':
            reward = (1-(abs(bid-self._action(self.wrtarget[self.gen_s(self.ctr_flag)], instance))/301.)) * (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)])
        elif self.dataset.dataset == 'tenmax':
            reward = (1-(abs(bid-self._action(self.wrtarget[self.gen_s(self.ctr_flag)], instance))/601.)) * (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)])
        reward *= (1.-self.alpha)
        if bid > wp:
            #reward += self.alpha*(1-wp/300.)* (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)]) #encourage buying
            self.cost += wp
            self.imp += 1
            self.budget -= wp
            '''if self.dataset.simple:
                reward = instance['ctr']
            else:
                reward = instance['lg_ctr']'''
            if self.retest and self.train == False and self.speclu == self.gen_s(self.ctr_flag):
                
                self.wpcount += 1.
                self.moniwp += wp
                '''if self.specount > 1000:
                    monitorwp = self.wpcount/self.specount
                    if monitorwp > self.wrtarget[self.gen_s(self.ctr_flag)]:
                        self.discount = max(monitorwp/self.wrtarget[self.gen_s(self.ctr_flag)], self.discount)
                '''
            if click:
                self.click += 1    
                #reward += self.alpha*(1-wp/300.) * (1. - self.model.clusters[3][self.gen_s(self.ctr_flag)])
                reward += self.alpha#1
                #reward += 1.#+ (1 - abs(cwr-cctr)) * (1-self.model.clusters[3][self.gen_s(self.ctr_flag)])#2.-abs(cwr-cctr)
            #else:
            #    reward = cctr - abs(cwr-cctr)
                #reward = 1. 
            #else:
                #reward = (-0.01)*self.get_reward()#(-1)*(bid/301)*(1-self.get_reward())
        #else:
            #reward = 1.-abs(cwr-cctr)
            #tempctr = self.get_reward()
            #reward = tempctr*(-0.01)+(1-tempctr)*0.001
            #    reward = max(-1.*bid/(self.budget+1e-7),-1)
        #elif bid <= wp and click:
        #    reward = -1.
        self.totalreward += reward
        s = self.gen_s(self.ctr_flag)
        '''if instance['ctr'] >= 0.05:
            s = 3
        elif instance['ctr'] < 0.05 and instance['ctr'] >= 0.005:
            s = 2
        else:
            s = 1
        '''
            #if bid > wp and reward != 1:
            #    reward = -1*wp/(self.cost/self.imp)
        if self.dataset.dataset == 'ipin':
            try:
                self.act_record.append([float(self.budget)/self.hourly_budget, self.remain_time, s, instance['timestamp'], adj, self.ctr_flag, act, wp, bid, click, self.remain_req, self.remain_req/self.total_req*1., self.instance['timestamp'].hour, cctr])
            except ZeroDivisionError:
                self.budget += 1e-7
                self.act_record.append([float(self.budget)/self.hourly_budget, self.remain_time, s, instance['timestamp'], adj, self.ctr_flag, act, wp, bid, click, self.remain_req, self.remain_req/self.total_req*1., self.instance['timestamp'].hour, cctr])
        elif self.dataset.dataset == 'tenmax':
            try:
                self.act_record.append([float(self.budget)/self.hourly_budget, self.remain_time, s, instance['dateTime'], adj, self.ctr_flag, act, wp, bid, click, self.remain_req, self.remain_req/self.total_req*1., self.instance['dateTime'].hour, cctr])
            except ZeroDivisionError:
                self.budget += 1e-7
                self.act_record.append([float(self.budget)/self.hourly_budget, self.remain_time, s, instance['dateTime'], adj, self.ctr_flag, act, wp, bid, click, self.remain_req, self.remain_req/self.total_req*1., self.instance['dateTime'].hour, cctr])
        '''while(random.random()<self.skip and self.remain_req > 0):
            sele.instance = self.dataset.next_record()
            self.remain_req -= 1
            self.ctr_flag = self.instance['ctr']
            if len(self.instance.keys()) == 0:
                print self.remain_req, 'loop'
                break
        '''
        self.instance, rcount = self.dataset.next_record()
        self.remain_time = self.dataset.rest_time(self.instance)
        #if len(self.instance.keys()) != 0:
        #    self.remain_req -= (rcount+1)
        #else:
        self.remain_req -= rcount
        #if self.dataset.simple:
        #    self.ctr_flag = self.instance['ctr']
        #else:
        #self.ctr_flag = 0#self.model.ctrmodel.predict_proba(self.model.data_prepare(self.instance, pow(2, 20), ctrtrain = True))[0][1]#self.instance['ctr']
        #self.ctr_flag = self.instance['ctr']
        if (len(self.instance.keys()) == 0 or self.budget < self.avgcpm or self.remain_time <= 0 or self.remain_req == 0) and (self.dataset.start < self.dataset.end-datetime.timedelta(hours=self.dataset.gaphour)):
            #print('budget remain: ', self.budget)
            #print("End Here", self.dataset.start, self.dataset.end)
            self.feeddata = True
            self.dataset.start += datetime.timedelta(hours=self.dataset.gaphour)
            if self.budget < max(self.hourly_budget*0.1, self.avgcpm):
                self.turnon = True
            else:
                self.turnon = False

        if self.feeddata:      ##############if the next hour has no record, here comes problems!!!!!!!!!
            self.dataset.get_data()
            self.budget += self.hourly_budget
            self.remain_req = self.dataset.cursor.count()#-1
            self.total_req = self.dataset.cursor.count()
            self.instance, rcount = self.dataset.next_record()
            while(len(self.instance) == 0):
                if (len(instance.keys()) == 0 or self.budget < self.avgcpm or self.remain_time <= 0 or self.remain_req == 0) and self.dataset.start >= self.dataset.end-datetime.timedelta(hours=self.dataset.gaphour):
                    self.end_episode = True
                    print("End Episode! ", self.dataset.start, self.dataset.end)
                    ob = self._get_state()
                    return ob, 0, self.end_episode, {}
                self.dataset.start += datetime.timedelta(hours=self.dataset.gaphour)
                self.dataset.get_data()
                self.budget += self.hourly_budget
                self.remain_req = self.dataset.cursor.count()#-1
                self.total_req = self.dataset.cursor.count()
                self.instance, rcount = self.dataset.next_record()
                self.remain_time = self.dataset.rest_time(self.instance)
            #print("Here to next hour", self.dataset.start, self.dataset.end, self.budget)
            self.remain_req -= rcount
            #self.wrtarget = self.cal_wr_b(self.avgreq, self.avgcpm)
            #self.remain_time = self.dataset.rest_time()
            #if self.dataset.simple:
            #    self.ctr_flag = self.instance['ctr']
            #else:
            #print("Here to next hour", self.dataset.start, self.dataset.end, self.budget)
            try:
                self.ctr_flag = self.model.ctrmodel.predict_proba(self.model.data_prepare(self.instance, pow(2, 20), ctrtrain = True))[0][1]#self.instance['ctr']
            except KeyError:
                print(self.instance)
                self.ctr_flag = 0#self.instance['lg_ctr']
            #self.ctr_flag = self.instance['ctr']
            #print("Here to next hour", self.dataset.start, self.dataset.end, self.budget)
            self.feeddata = False
        '''if (self.budget == 0 or self.remain_time <= 0 or self.remain_req == 0) and self.dataset.start == self.dataset.end-datetime.timedelta(hours=self.dataset.gaphour):
            self.end_episode = True
            print("End Episode! \n", self.dataset.start)
            ob = self._get_state()
            return ob, 0, self.end_episode, {}
        '''
        #self.remain_req -= 1
        if len(self.instance.keys()) == 0:
            if (self.budget < self.avgcpm or self.remain_time <= 0 or self.remain_req == 0) and self.dataset.start == self.dataset.end-datetime.timedelta(hours=self.dataset.gaphour):
                self.end_episode = True
                print("End Episode! \n", self.dataset.start)
                ob = self._get_state()
                return ob, reward, self.end_episode, {}
        else:
            self.ctr_flag = self.model.ctrmodel.predict_proba(self.model.data_prepare(self.instance, pow(2, 20), ctrtrain = True))[0][1]
        ob = self._get_state()
        #self.act_record.append([130./float(self.budget), self.remain_time, act, instance['timestamp'], instance['bid_id']])
        return ob, reward, self.end_episode, {}
    
    def _render(self, mode='human', close=False):


        pass    

    def reset(self):
        print("reset")
        self.get_final_result()
        #self.total_budget = random.randint(52000, 5000000)
        if self.train:
            self.total_budget = self.test_budget*(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)
            #self.total_budget = random.randrange(20000, 250000)*(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)
 
            #self.total_budget = np.random.choice([20000., 50000., 250000., 500000., 1000000., 1500000.], 1)[0]*(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)#random.randrange(20000, 250000)*(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)
            self.budget = self.total_budget/(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)
            self.hourly_budget = self.total_budget/(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)
            if self.budget < 0:
                self.budget = self.test_budget*(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)
                self.hourly_budget = self.test_budget#/(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)
        else:
            self.total_budget = self.test_budget*(((self.realend-self.realstart).total_seconds()/3600)/self.dataset.gaphour)
            self.budget = self.test_budget#self.hourly_budget
            self.hourly_budget = self.test_budget#/(((self.dataset.end-self.dataset.start).total_seconds()/3600)/self.dataset.gaphour)
        self.current_buget = self.budget
        #if self.train:
        self.dataset.start = self.realstart
        self.dataset.end = self.realend

        self.dataset.get_data()
        self.remain_time = 1
        self.remain_req = self.dataset.cursor.count()#-1 
        self.total_req = self.dataset.cursor.count()
        self.end_episode = False
        self.instance, count = self.dataset.next_record()
        #if len(self.instance.keys()) == 0:
        #    self.remain_req += 1
        self.remain_req -= count
        #if self.dataset.simple:
        #    self.ctr_flag = self.instance['ctr']
        #else:
        self.ctr_flag = self.model.ctrmodel.predict_proba(self.model.data_prepare(self.instance, pow(2, 20), ctrtrain = True))[0][1]#self.instance['ctr']
        self.speclu = -1
        self.wrtarget = self.cal_wr_b(self.avgreq, self.avgcpm)#(13300, 178.)#22000, 150.)
        self.episode = 0
        self.click = 0
        self.imp = 0
        self.cost = 0.
        self.bid = 0.
        self.totalreward = 0.
        self.feeddata = False  
        
        self.turnon = False 
        self.count = 0.
        self.specount = 0.
        #self.speclu = -1
        self.moniwp = 0.
        self.wpcount = 0.
        self.discount = 0.
        self.act_record = []
        print('Reset: ', self.budget, self.remain_time)
        return self._get_state()
    def gen_s(self, ctr):
        for j in range(len(self.model.clusters[0])):
            if ctr >= self.model.clusters[0][j]:
                return self.model.clusters[1][j]
    '''def gen_s(self, ctr):
        if ctr >= 0.05:
            s = 5
        elif ctr <0.05 and ctr >= 0.03:
            s = 4
        elif ctr < 0.03 and ctr >= 0.01:
            s = 3
        elif ctr < 0.01 and ctr >= 0.005:
            s = 2
        elif ctr < 0.005 and ctr >= 0.00075:
            s = 1
        else:
            s = 0
        return s
    '''
    def _get_state(self):
        clust = [0]*8
        s = self.gen_s(self.ctr_flag)
        clust[s] = 1
        self.remain_time = self.dataset.rest_time(self.instance)
        if self.remain_req > 0:
            req = self.remain_req/self.total_req
        else:
            req = 0.
        hrs = [0]*24
        if len(self.instance.keys()) != 0:
            if self.dataset.dataset == 'ipin':
                hr = self.instance['timestamp'].hour
            elif self.dataset.dataset == 'tenmax':
                hr = self.instance['dateTime'].hour
        else:
            hr = 0
        hrs[hr] = 1
        time = [0]*11
        time[int(math.floor(10*self.remain_time))] = 1
        #ob = [self.budget/float(self.current_budget), self.remain_time, 0]#self.remain_req]
        #ob = [130./self.budget, self.remain_time, req, hr, s]#self.remain_req]
        #ob = [math.floor(math.log10(max(self.budget, 0)))/10., math.floor(10*self.remain_time)/10.]
        #ob = [self.avgcpm/self.budget]#, math.floor(10*self.remain_time)/10.]
        ob = [self.remain_time]
        '''a = self.budget
        for i in range(6, -1, -1):
            x, y = divmod(a, pow(10, i))
            a = y
            ob.append(x)'''
        #ob.extend(time)
        ob.extend(hrs)
        ob.extend(clust)
        return ob
    def get_reward(self):
        return float(self.model.clusters[2][self.gen_s(self.ctr_flag)][1])

    def cal_wr_b(self, avg_req, avg_cpm):
        if self.dataset.aid == '1458':
            ratio = 0.961#0.79#0.98
        elif self.dataset.aid == '3386':
            ratio = 0.717#0.741#0.725
        elif self.dataset.aid == '215':
            ratio = 0.634#0.99#0.66
        ratio = 0.5 
        k = len(self.model.clusters[2])
        m = [0.]*k
        temp = 0.
        thre = self.budget#/(avg_req*avg_cpm)
        clu_order = []
        import copy
        tempclucen = copy.deepcopy(self.model.clusters[2])
        for cck in range(0, k):
            maxc = -1
            maxcv = -1
            for cckk in range(0, len(self.model.clusters[2])):
                if float(tempclucen[cckk][1])*(1-ratio)+float(tempclucen[cckk][0])*ratio > maxcv:
                    maxc = cckk
                    maxcv = float(tempclucen[cckk][1])*(1-ratio)+float(tempclucen[cckk][0])*ratio#float(tempclucen[cckk][1])
            clu_order.append(maxc)
            print(maxc, tempclucen[maxc][0], tempclucen[maxc][1], float(tempclucen[maxc][1])*(1-ratio)+float(tempclucen[maxc][0])*ratio)
            tempclucen[maxc][1] = -1
            tempclucen[maxc][0] = -1  
            
        print('order', self.model.clusters[1], clu_order)
        for i in clu_order:#self.model.clusters[1][:8]:
            temp += self.model.clusters[3][i]*avg_req*self.model.clusters[4][i][self.pricetype] #change here for cluster price
            if temp > thre:
                m[i] = 1. - (temp - thre) / (self.model.clusters[3][i]*avg_req*self.model.clusters[4][i][self.pricetype]) #change here for cluster price
                self.speclu = i
                break
            else:
                m[i] = 1.
        print(m)
        return m
    def get_final_result(self):
        print("Budget: ", self.budget, "Cost: ", self.cost, "Clicks: ", self.click, "Impressions: ", self.imp)
        #if self.train == 0:
        #    with open('log/Result.txt', 'a+') as a:
        #        a.write("Budget: {0}, Cost: {1}, Clicks: {2}, Imps: {3}\n".format(self.budget, self.cost, self.click, self.imp))

    def record_result(self, gpu):
        if self.train == 0:
            with open('log/{0}Result.txt'.format(gpu), 'a+') as a:
                a.write("Aid: {0}, Hourly Budget: {1}, PriceType: {2}, Alpha: {3}".format( self.dataset.aid, self.hourly_budget, self.pricetype, self.alpha))
                a.write("Budget: {0}, Cost: {1}, Clicks: {2}, Imps: {3}, TotalRewards: {4}\n".format(self.budget, self.cost, self.click, self.imp, self.totalreward))
