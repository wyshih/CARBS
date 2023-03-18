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

    def __init__(self, dates, cm, bu, gaphour = 1, appro = '1', aid = None):
        
        
        self.name = "Advertiser_A"

        self.appro = appro

        self.lifetime = 96/gaphour#271#(dates-1)*24 #except the market observation of first day

        self.timeslot = 0

        self.budget = bu#1000000#5900000#123587600#5900000

        self.remain_totalbudget = self.budget

        self.ad_performance ="branding"#"performance"

        self.avg_budget = float(self.budget)/self.lifetime #constant

        self.budget_plan = [1./self.lifetime]*int(self.lifetime)
        '''if gaphour == 1:
            if aid == None:
                self.budget_plan = [0.01352648304852096, 0.0037818807938874966, 0.0037894302800831926, 0.004101567218428873, 0.001953257973904583, 0.002087364301417217, 0.003557180632026513, 0.005016290418576458, 0.006833383114187763, 0.008770855798774074, 0.01381789321567482, 0.014197426476240256, 0.015029517118391317, 0.011663544382193974, 0.01107344908846131, 0.011281403117306387, 0.02181513257446813, 0.016851002242334663, 0.014667004517612539, 0.014381222149259108, 0.014700084993488224, 0.013281742432395037, 0.015560177366256048, 0.008262706746111053, 0.01352648304852096, 0.0037818807938874966, 0.0037894302800831926, 0.004101567218428873, 0.001953257973904583, 0.002087364301417217, 0.003557180632026513, 0.005016290418576458, 0.006833383114187763, 0.008770855798774074, 0.01381789321567482, 0.014197426476240256, 0.015029517118391317, 0.011663544382193974, 0.01107344908846131, 0.011281403117306387, 0.02181513257446813, 0.016851002242334663, 0.014667004517612539, 0.014381222149259108, 0.014700084993488224, 0.013281742432395037, 0.015560177366256048, 0.008262706746111053, 0.01352648304852096, 0.0037818807938874966, 0.0037894302800831926, 0.004101567218428873, 0.001953257973904583, 0.002087364301417217, 0.003557180632026513, 0.005016290418576458, 0.006833383114187763, 0.008770855798774074, 0.01381789321567482, 0.014197426476240256, 0.015029517118391317, 0.011663544382193974, 0.01107344908846131, 0.011281403117306387, 0.02181513257446813, 0.016851002242334663, 0.014667004517612539, 0.014381222149259108, 0.014700084993488224, 0.013281742432395037, 0.015560177366256048, 0.008262706746111053, 0.01352648304852096, 0.0037818807938874966, 0.0037894302800831926, 0.004101567218428873, 0.001953257973904583, 0.002087364301417217, 0.003557180632026513, 0.005016290418576458, 0.006833383114187763, 0.008770855798774074, 0.01381789321567482, 0.014197426476240256, 0.015029517118391317, 0.011663544382193974, 0.01107344908846131, 0.011281403117306387, 0.02181513257446813, 0.016851002242334663, 0.014667004517612539, 0.014381222149259108, 0.014700084993488224, 0.013281742432395037, 0.015560177366256048, 0.008262706746111053]
            elif aid == '1458':
                self.budget_plan = [0.010285656719425105, 0.006222809256232883, 0.0030030147678852597, 0.00355951554769527, 0.001657562664636922, 0.0020926145649107883, 0.0023409224891237025, 0.004732775153536755, 0.006947211712820972, 0.008945801338736036, 0.013608617460281925, 0.01389162506436231, 0.014356339594200303, 0.013889386375338602, 0.014544016357354467, 0.013972777541471713, 0.014728708201810353, 0.014415291738491274, 0.014112322490616162, 0.01443842485840292, 0.014605953420343713, 0.014295148760885626, 0.015560194616699127, 0.013793309304737811, 0.010285656719425105, 0.006222809256232883, 0.0030030147678852597, 0.00355951554769527, 0.001657562664636922, 0.0020926145649107883, 0.0023409224891237025, 0.004732775153536755, 0.006947211712820972, 0.008945801338736036, 0.013608617460281925, 0.01389162506436231, 0.014356339594200303, 0.013889386375338602, 0.014544016357354467, 0.013972777541471713, 0.014728708201810353, 0.014415291738491274, 0.014112322490616162, 0.01443842485840292, 0.014605953420343713, 0.014295148760885626, 0.015560194616699127, 0.013793309304737811, 0.010285656719425105, 0.006222809256232883, 0.0030030147678852597, 0.00355951554769527, 0.001657562664636922, 0.0020926145649107883, 0.0023409224891237025, 0.004732775153536755, 0.006947211712820972, 0.008945801338736036, 0.013608617460281925, 0.01389162506436231, 0.014356339594200303, 0.013889386375338602, 0.014544016357354467, 0.013972777541471713, 0.014728708201810353, 0.014415291738491274, 0.014112322490616162, 0.01443842485840292, 0.014605953420343713, 0.014295148760885626, 0.015560194616699127, 0.013793309304737811, 0.010285656719425105, 0.006222809256232883, 0.0030030147678852597, 0.00355951554769527, 0.001657562664636922, 0.0020926145649107883, 0.0023409224891237025, 0.004732775153536755, 0.006947211712820972, 0.008945801338736036, 0.013608617460281925, 0.01389162506436231, 0.014356339594200303, 0.013889386375338602, 0.014544016357354467, 0.013972777541471713, 0.014728708201810353, 0.014415291738491274, 0.014112322490616162, 0.01443842485840292, 0.014605953420343713, 0.014295148760885626, 0.015560194616699127, 0.013793309304737811]
            elif aid == '3386':
                self.budget_plan = [0.015686883052228703, 0.0034410742541135138, 0.0034046083402301474, 0.003893330859993087, 0.002513571661862679, 0.0019796631183746985, 0.001905145816091298, 0.00403344717040363, 0.007375032898596221, 0.007659229205708977, 0.013044887954515907, 0.01364380095318728, 0.013785700922429074, 0.015337682289932554, 0.013801952036224922, 0.014599643902423557, 0.014927638943059268, 0.013959508566318814, 0.014432376340915072, 0.014968464911863471, 0.013062328174199256, 0.01523700465812413, 0.01678898602562761, 0.010518037943576133, 0.015686883052228703, 0.0034410742541135138, 0.0034046083402301474, 0.003893330859993087, 0.002513571661862679, 0.0019796631183746985, 0.001905145816091298, 0.00403344717040363, 0.007375032898596221, 0.007659229205708977, 0.013044887954515907, 0.01364380095318728, 0.013785700922429074, 0.015337682289932554, 0.013801952036224922, 0.014599643902423557, 0.014927638943059268, 0.013959508566318814, 0.014432376340915072, 0.014968464911863471, 0.013062328174199256, 0.01523700465812413, 0.01678898602562761, 0.010518037943576133, 0.015686883052228703, 0.0034410742541135138, 0.0034046083402301474, 0.003893330859993087, 0.002513571661862679, 0.0019796631183746985, 0.001905145816091298, 0.00403344717040363, 0.007375032898596221, 0.007659229205708977, 0.013044887954515907, 0.01364380095318728, 0.013785700922429074, 0.015337682289932554, 0.013801952036224922, 0.014599643902423557, 0.014927638943059268, 0.013959508566318814, 0.014432376340915072, 0.014968464911863471, 0.013062328174199256, 0.01523700465812413, 0.01678898602562761, 0.010518037943576133, 0.015686883052228703, 0.0034410742541135138, 0.0034046083402301474, 0.003893330859993087, 0.002513571661862679, 0.0019796631183746985, 0.001905145816091298, 0.00403344717040363, 0.007375032898596221, 0.007659229205708977, 0.013044887954515907, 0.01364380095318728, 0.013785700922429074, 0.015337682289932554, 0.013801952036224922, 0.014599643902423557, 0.014927638943059268, 0.013959508566318814, 0.014432376340915072, 0.014968464911863471, 0.013062328174199256, 0.01523700465812413, 0.01678898602562761, 0.010518037943576133]
        elif gaphour == 6:
            if aid == None:
                self.budget_plan = [0.02860835746469603, 0.05214010626363315, 0.0924717488723405, 0.07677978739933032, 0.02860835746469603, 0.05214010626363315, 0.0924717488723405, 0.07677978739933032, 0.02860835746469603, 0.05214010626363315, 0.0924717488723405, 0.07677978739933032, 0.02860835746469603, 0.05214010626363315, 0.0924717488723405, 0.07677978739933032]
            elif aid == '1458':
                self.budget_plan = [0.026821173520786227, 0.0504669532188617, 0.08590651980866672, 0.08680535345168536, 0.026821173520786227, 0.0504669532188617, 0.08590651980866672, 0.08680535345168536, 0.026821173520786227, 0.0504669532188617, 0.08590651980866672, 0.08680535345168536, 0.026821173520786227, 0.0504669532188617, 0.08590651980866672, 0.08680535345168536]
            elif aid == '3386':
                self.budget_plan = [0.030919131286802827, 0.047661543998503315, 0.0864121266603882, 0.08500719805430568, 0.030919131286802827, 0.047661543998503315, 0.0864121266603882, 0.08500719805430568, 0.030919131286802827, 0.047661543998503315, 0.0864121266603882, 0.08500719805430568, 0.030919131286802827, 0.047661543998503315, 0.0864121266603882, 0.08500719805430568]'''
        self.current_status = self.budget_plan[0]*self.budget#float(self.budget)/self.lifetime
    
        self.next_spend =  0#self.avg_budget+(self.remain_totalbudget-self.avg_budget*(self.lifetime-self.timeslot))/(self.lifetime-self.timeslot)
        self.spend = 0

        self.pacing_rate = 1.0 # for our method

        self.trial_rate = 0.001
        self.global_pacingrate = 0.3 # for smart pacing
        if appro == '1' or appro == '3':
            self.eps = 0
        else:
            self.eps = 1.0#percentage
        
        self.cpc = 18750.
        self.unit_click = int(self.current_status/self.cpc) + 1
        self.cpm = cm#9924.
        self.ecpc = 33207.
        self.layers = [Layer() for i in range(0, int(1/self.global_pacingrate))]
        self.init_layer()
        self.stat = [50.]        
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
            print (up/down )
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

    def cal_winrate(self, down, up, current_status, coef, feature_size, pacing_rate, requests, index, betax, eps, stat, eps_coef = None, eps_betax = None):
        price_coef = coef[feature_size]
        if current_status <= 0:
            return 0.001
        if requests <= 0:    #underestimate request
            requests = 1
        #if current_status - stat*(requests-1) < 0:
        #    current_status = stat
        #current_status /= requests
        #requests = 1.
        #print index, current_status, price_coef, betax, pacing_rate, requests
        try:
            #wof = (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax)/(pacing_rate*requests)) #origin
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
            print (index, current_status, price_coef, betax, pacing_rate, requests)
            print ((-1)*current_status*price_coef/(pacing_rate*requests)-betax)
            print (math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax))
            print (current_status*price_coef*math.exp((-1)*current_status*price_coef/(pacing_rate*requests)-betax))
            print ((pacing_rate*requests))
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
    
