import pickle, os, sys, timeit, random, math

sys.path.append('/home/mpc/RTB/schema/')

import itertools, gc

import numpy as np
import scipy
from scipy.stats import lognorm
from scipy.sparse import vstack, hstack, csr_matrix, coo_matrix
from scipy.special import lambertw
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.preprocessing import scale, normalize
#from sklearn.cross_validation import ShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
class Layer():

    def __init__(self):

        self.avg_performance = 0.
        self.pri = 0.
        self.p_r = 0.001
        self.old_p_r = 0.
        self.ecpc = 0.
        self.total_spend = list()
        self.spend = 0.
        self.next_spend = 0
class ADer:

    def __init__(self, dates, cm=300, bu = 1000000, gaphour = 1, appro = '1', aid = None):
        
        
        self.name = "Advertiser_A"
    
        self.lifetime = int(96/gaphour)#271#(dates-1)*24 #except the market observation of first day

        self.appro = appro

        self.timeslot = 0

        self.budget = bu#1000000#5900000#123587600#5900000

        self.remain_totalbudget = self.budget

        self.ad_performance ="branding"#"performance"
        self.stat = [50.]
        self.avg_budget = float(self.budget)/self.lifetime #constant
        '''if gaphour == 1:
            if aid == None:
                self.budget_plan = [0.009060358996891732, 0.004562726814106007, 0.00302079773470683, 0.0018816416866018976, 0.0011940796432814203, 0.0012815505541180492, 0.002524451170889681, 0.004086722322576446, 0.007703542775309606, 0.008315839151166007, 0.01114542140636951, 0.012298816905075755, 0.014158082312161304, 0.013293544239938811, 0.01247782714120653, 0.01037649107389868, 0.009304463864342789, 0.012103533011114909, 0.014776481309703982, 0.017811518495012123, 0.020458022099627333, 0.01910934270696024, 0.019963709743038942, 0.019091034841901413, 0.009060358996891732, 0.004562726814106007, 0.00302079773470683, 0.0018816416866018976, 0.0011940796432814203, 0.0012815505541180492, 0.002524451170889681, 0.004086722322576446, 0.007703542775309606, 0.008315839151166007, 0.01114542140636951, 0.012298816905075755, 0.014158082312161304, 0.013293544239938811, 0.01247782714120653, 0.01037649107389868, 0.009304463864342789, 0.012103533011114909, 0.014776481309703982, 0.017811518495012123, 0.020458022099627333, 0.01910934270696024, 0.019963709743038942, 0.019091034841901413, 0.009060358996891732, 0.004562726814106007, 0.00302079773470683, 0.0018816416866018976, 0.0011940796432814203, 0.0012815505541180492, 0.002524451170889681, 0.004086722322576446, 0.007703542775309606, 0.008315839151166007, 0.01114542140636951, 0.012298816905075755, 0.014158082312161304, 0.013293544239938811, 0.01247782714120653, 0.01037649107389868, 0.009304463864342789, 0.012103533011114909, 0.014776481309703982, 0.017811518495012123, 0.020458022099627333, 0.01910934270696024, 0.019963709743038942, 0.019091034841901413, 0.009060358996891732, 0.004562726814106007, 0.00302079773470683, 0.0018816416866018976, 0.0011940796432814203, 0.0012815505541180492, 0.002524451170889681, 0.004086722322576446, 0.007703542775309606, 0.008315839151166007, 0.01114542140636951, 0.012298816905075755, 0.014158082312161304, 0.013293544239938811, 0.01247782714120653, 0.01037649107389868, 0.009304463864342789, 0.012103533011114909, 0.014776481309703982, 0.017811518495012123, 0.020458022099627333, 0.01910934270696024, 0.019963709743038942, 0.019091034841901413]#gaponehour 
            elif aid == '215':
                self.budget_plan = [0.008821885117319182, 0.006043272921464571, 0.004007292813991523, 0.00242815660358466, 0.0021381963218851173, 0.0017042989219318448, 0.0024511031006975736, 0.004103250892827343, 0.007090467607890258, 0.00908889890190581, 0.012439087480391174, 0.014059944594639698, 0.013609358833149761, 0.01342995894663062, 0.01200936217082207, 0.012395280531357432, 0.010547044491171857, 0.011754864657387938, 0.01206568539100831, 0.01380753312639765, 0.01660700577417309, 0.01678431961550015, 0.022308167284136045, 0.020305563899736322, 0.008821885117319182, 0.006043272921464571, 0.004007292813991523, 0.00242815660358466, 0.0021381963218851173, 0.0017042989219318448, 0.0024511031006975736, 0.004103250892827343, 0.007090467607890258, 0.00908889890190581, 0.012439087480391174, 0.014059944594639698, 0.013609358833149761, 0.01342995894663062, 0.01200936217082207, 0.012395280531357432, 0.010547044491171857, 0.011754864657387938, 0.01206568539100831, 0.01380753312639765, 0.01660700577417309, 0.01678431961550015, 0.022308167284136045, 0.020305563899736322, 0.008821885117319182, 0.006043272921464571, 0.004007292813991523, 0.00242815660358466, 0.0021381963218851173, 0.0017042989219318448, 0.0024511031006975736, 0.004103250892827343, 0.007090467607890258, 0.00908889890190581, 0.012439087480391174, 0.014059944594639698, 0.013609358833149761, 0.01342995894663062, 0.01200936217082207, 0.012395280531357432, 0.010547044491171857, 0.011754864657387938, 0.01206568539100831, 0.01380753312639765, 0.01660700577417309, 0.01678431961550015, 0.022308167284136045, 0.020305563899736322, 0.008821885117319182, 0.006043272921464571, 0.004007292813991523, 0.00242815660358466, 0.0021381963218851173, 0.0017042989219318448, 0.0024511031006975736, 0.004103250892827343, 0.007090467607890258, 0.00908889890190581, 0.012439087480391174, 0.014059944594639698, 0.013609358833149761, 0.01342995894663062, 0.01200936217082207, 0.012395280531357432, 0.010547044491171857, 0.011754864657387938, 0.01206568539100831, 0.01380753312639765, 0.01660700577417309, 0.01678431961550015, 0.022308167284136045, 0.020305563899736322]
        elif gaphour == 6:
            if aid == None:
                self.budget_plan = [0.021001155429705935, 0.046074793731387, 0.07171394164266302, 0.11121010919624404, 0.021001155429705935, 0.046074793731387, 0.07171394164266302, 0.11121010919624404, 0.021001155429705935, 0.046074793731387, 0.07171394164266302, 0.11121010919624404, 0.021001155429705935, 0.046074793731387, 0.07171394164266302, 0.11121010919624404] #6hours 
            elif aid == '215':
                self.budget_plan = [0.025143102700176896, 0.04923275257835186, 0.07374586963051967, 0.10187827509095157, 0.025143102700176896, 0.04923275257835186, 0.07374586963051967, 0.10187827509095157, 0.025143102700176896, 0.04923275257835186, 0.07374586963051967, 0.10187827509095157, 0.025143102700176896, 0.04923275257835186, 0.07374586963051967, 0.10187827509095157]
        '''
        #else:'''
        self.budget_plan = [1./self.lifetime]*self.lifetime

        self.current_status = self.budget_plan[0]*self.budget#float(self.budget)/self.lifetime
    
        self.next_spend =  0#self.avg_budget+(self.remain_totalbudget-self.avg_budget*(self.lifetime-self.timeslot))/(self.lifetime-self.timeslot)
        self.spend = 0

        self.pacing_rate = 1.0 # for our method

        self.trial_rate = 0.001
        self.global_pacingrate = 0.3 # for smart pacing

        #self.eps = 0
        #self.eps = 1.0#percentage
        if appro == '1' or appro == '3':
            self.eps = 0
        else:
            self.eps = 1.0

        self.cpc = 18750.
        self.unit_click = int(self.current_status/self.cpc) + 1
        self.cpm = cm#9924.
        self.ecpc = 33207.
        self.layers = [Layer() for i in range(0, int(1/self.global_pacingrate))]
        self.init_layer()
        
        self.costs = 0.
        self.clicks = 0
        self.imps = 0.
        
    def init_layer(self):
        
        for i in range(0, len(self.layers)):
            self.layers[i].p_r = self.global_pacingrate
            self.layers[i].old_p_r = self.global_pacingrate
    def adj(self, performance, goal = 0):
        self.next_spend =  self.avg_budget+(self.remain_totalbudget-self.avg_budget*(self.lifetime-self.timeslot))/(self.lifetime-self.timeslot)
        R = self.next_spend-self.spend

        if R == 0:
            print ("No adjustment!")
        elif R > 0:
            for each in range(len(self.layers)-1, 0, -1):
                if self.layers[each].spend == 0:
                    self.layers[each].spend += 0.000001
                if self.layers[each].p_r == 0 and each != len(self.layers)-1:
                    if self.layers[each+1].p_r > self.trial_rate:
                        self.layers[each].p_r = self.trial_rate
                    break
                self.layers[each].old_p_r = self.layers[each].p_r
                self.layers[each].p_r = min(1.0, self.layers[each].p_r*(self.layers[each].spend+R)/self.layers[each].spend)
                #print 'more: ', each, self.layers[each].old_p_r, self.layers[each].p_r  
                R = R - self.layers[each].spend*(self.layers[each].p_r-self.layers[each].old_p_r)/self.layers[each].old_p_r
                
        else:
            for each in range(0, len(self.layers)):
                if self.layers[each].spend == 0:
                    self.layers[each].spend += 0.000001
 
                if self.layers[each].p_r == 0:
                    continue

                self.layers[each].old_p_r = self.layers[each].p_r

                self.layers[each].p_r = max(0.0, self.layers[each].old_p_r*(self.layers[each].spend+R)/self.layers[each].spend)
                R = R-self.layers[each].spend*(self.layers[each].p_r-self.layers[each].old_p_r)/self.layers[each].old_p_r
                if R >= 0:
                    if each != 0 and self.layers[each].p_r > self.trial_rate:
                        self.layers[each-1].p_r = self.trial_rate
                #print 'less: ', each, self.layers[each].old_p_r, self.layers[each].p_r
        if performance == "performance":
            if self.expperf(0) > self.ecpc:
                for each in range(0, len(self.layers)): #should len(self.layers)-1
                    if self.expperf(each+1) > self.ecpc:
                        self.layers[each].p_r = 0.0
                    else:
                        temp = 0.
                        for m in range(each+1, len(self.layers)):
                            temp += self.layers[m].spend*(self.ecpc/self.layers[m].ecpc-1)
                        self.layers[each].p_r = self.layers[each].old_p_r * temp/(self.layers[each].spend*(1-self.ecpc/self.layers[each].ecpc))
                        if each != 0:
                            self.layers[each-1].p_r = self.trial_rate
                        break

        for each in range(0, len(self.layers)):
            self.layers[each].total_spend.append(self.layers[each].spend)
            self.layers[each].spend = 0
        self.timeslot += 1
    def bidding_smart_first(self, ctr, winp):
        buy = list()
        cost = list()
        for i in range(0, len(winp)):
            if self.cpm/1000 > winp[i] and random.random() <= self.global_pacingrate:
                buy.append(ctr[i])
                cost.append(winp[i])
        if len(buy) == 0:
            print ("not buying")
            return 0
        buy.sort()
        length = len(buy)/len(self.layers)
        for i in range(0, len(self.layers)):
            self.layers[i].spend = float(sum(cost[i*length:(i+1)*length]))/len(cost[i*length:(i+1)*length])
            self.layers[i].avg_performance = float(sum(buy[i*length:(i+1)*length]))/len(buy[i*length:(i+1)*length])
            self.layers[i].ecpc = self.cpm/(1000*self.layers[i].avg_performance)
            self.layers[i].total_spend.append(float(sum(cost[i*length:(i+1)*length]))/len(cost[i*length:(i+1)*length]))
            #if i >= len(self.layers)/2:
            #    self.layers[i].p_r = 1.0
            #else:
            #    self.layers[i].p_r = 0.0
        self.remain_totalbudget -= sum(cost)#self.avg_budget
        self.next_spend =  self.avg_budget+(self.remain_totalbudget-self.avg_budget*(self.lifetime-self.timeslot))/(self.lifetime-self.timeslot)
        self.timeslot += 1

    def bidding_smart(self, ctr, winp, cli = None):
        imps = []
        clis = []
        for i in range(0, len(ctr)):
            index = self.find_group(ctr[i])
            if self.cpm/1000 > winp[i] and random.random() <= self.layers[index].p_r:
                self.remain_totalbudget -= winp[i]
                self.layers[index].spend += winp[i]
                imps.append(winp[i])
                self.imps += 1
                self.costs += winp[i]
                if cli != None:
                    if cli[i] == True:
                        clis.append(True)
                        self.clicks += 1
                    else:
                        clis.append(False)
        self.spend = sum(imps)
        '''
        if cli != None:
            print "Imps: ", len(imps), sum(imps), sum(clis)
        else:
            print "Imps: ", len(imps), sum(imps)
        '''
    def find_group(self, ctr):
        index = -1
        temp = 1.
        for i in range(0, len(self.layers)):
            if abs(ctr-self.layers[i].avg_performance) <= temp:
                temp = abs(ctr-self.layers[i].avg_performance)
                index = i
            if i != len(self.layers)-1 and abs(ctr-self.layers[i].avg_performance)<abs(ctr-self.layers[i+1].avg_performance):
                break
        return index
        
            
    def expperf(self, i):
        up = 0.
        down = 0.
        for ind in range(i, len(self.layers)):
            up += self.layers[ind].spend*self.layers[ind].p_r/self.layers[ind].old_p_r
            down += self.layers[ind].spend*self.layers[ind].p_r/(self.layers[ind].old_p_r*self.layers[ind].ecpc)  
        try: 
            print (up/down)
        except: 
            print (up, self.layers[ind].spend, self.layers[ind].p_r, self.layers[ind].old_p_r, self.layers[ind].ecpc)
        return up/down
            
#=============================================the belowing functions are for my own method=====================================================
    def budget_next(self, rest):
        if self.timeslot < len(self.budget_plan)-1:
            self.timeslot += 1
            return self.budget_plan[self.timeslot]*self.budget+rest
        else:
            return rest
        #return float(self.budget)/self.lifetime+rest

    def budget_update(self, used):

        self.current_status -= used

    def cal_winrate(self, down, up, current_status, coef, feature_size, pacing_rate, requests, index, betax, eps, stat = None, eps_coef = None, eps_betax = None):
        price_coef = coef[feature_size]
        if current_status <= 0:
            return 0.001
        if requests <= 0:    #underestimate request
            requests = 1
        #print index, current_status, price_coef, betax, pacing_rate, requests
        try:
            #wof = (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax)/(pacing_rate*requests)) #origin
            #wof = (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax-price_coef*eps)/(pacing_rate*requests))   #with eps
            #wof = (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests*eps)-betax)/(pacing_rate*requests*eps))#percentage
            if self.appro == '1':
                wof = (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax-price_coef*eps)/(pacing_rate*requests))   #with eps
            elif self.appro == '3':
                wof = (-1.)*(price_coef*current_status*pow(math.exp((-1)*price_coef*current_status/(pacing_rate*requests)-eps_betax*price_coef+eps_coef*betax-betax), (-1.)/(eps_coef-1.)))/((eps_coef-1.)*pacing_rate*requests)
            elif self.appro == '2':
                #wof = (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax-price_coef*eps)/(pacing_rate*requests))   #with eps with stat
                wof = (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests*eps)-betax)/(pacing_rate*requests*eps))#percentage
            elif self.appro == '4':

                wof = (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests*eps)-betax)/(pacing_rate*requests*eps))#percentage

        except:
            print ((index, current_status, price_coef, betax, pacing_rate, requests))
            print ((-1)*current_status*price_coef/(pacing_rate*requests)-betax)
            print (math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax))
            print (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax))
            print (pacing_rate*requests)
            return 0.001
        #wr = (current_status*price_coef)/(pacing_rate*requests*lambertw(wof)+current_status*price_coef)
        #wr = (current_status*price_coef)/(eps*pacing_rate*requests*lambertw(wof)+current_status*price_coef)#percentage
        if self.appro == '1':
            wr = (current_status*price_coef)/(pacing_rate*requests*lambertw(wof)+current_status*price_coef)
        elif self.appro == '3':
            wr = (-1)*(price_coef*current_status)/((eps_coef-1)*pacing_rate*requests*(lambertw(wof)+price_coef*current_status/((1-eps_coef)*pacing_rate*requests)))
        elif self.appro == '2':
            wr = (current_status*price_coef)/(eps*pacing_rate*requests*lambertw(wof)+current_status*price_coef)#percentage
        elif self.appro == '4':
            wr = (current_status*price_coef)/(eps*pacing_rate*requests*lambertw(wof)+current_status*price_coef)#percentage

        if wr < down:
            wr = down
        if wr > up:
            wr = up
        return wr#random.uniform(up, down)    

    def cal_winrate_performance(self, down, up, current_status, coef, feature_size, pacing_rate, requests, index, betax, eps, ctr):
        price_coef = coef[feature_size]
        if current_status <= 0:
            return 0.001
        #print index, current_status, price_coef, betax, pacing_rate, requests
        try:
            wr = 1./(1+ math.exp((-1)*(price_coef*(ctr*self.cpc+eps)+betax)))
            #wof = price_coef*ctr*self.cpc*math.exp((-1)*price_coef*self.cpc*ctr-price_coef*eps-betax)
            
            #wr = price_coef*ctr*self.cpc/(lambertw(wof)+price_coef*ctr*self.cpc)

        except:
            #print 1/math.exp((-1)*(price_coef*(ctr*self.cpc+eps)+betax)), '---****---error'
            print (price_coef*ctr*self.cpc,  price_coef*ctr*self.cpc*math.exp((-1)*price_coef*self.cpc*ctr-price_coef*eps-betax), lambertw(wof), (lambertw(wof)+price_coef*ctr*self.cpc))
        if wr < down:
            wr = down
        if wr > up:
            wr = up
        return wr#random.uniform(up, down)
    
