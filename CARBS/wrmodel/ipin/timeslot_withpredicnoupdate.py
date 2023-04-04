import pickle, os, sys, timeit, random, math
from copy import deepcopy
import itertools, gc
from scipy.stats import norm
import numpy as np
import scipy
from scipy.stats import lognorm
from scipy.sparse import vstack, hstack, csr_matrix, coo_matrix
from scipy import sparse
from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDRegressor
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.preprocessing import scale, normalize

from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
import matplotlib.pyplot as plt

import onoff, advertiser
import warnings                                                                                                   
warnings.filterwarnings("ignore", category=DeprecationWarning)

dates = [("06","06"), ("06","07"), ("06","08"), ("06","09"), ("06","10"), ("06","11"), ("06","12")]#,

hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]


approxi = 3 
    

def main(noise, cm, bu, aid= None, gaphour = 1, th = .999, appro = '1', ct = 0.): 
    
    var_result = []
    slope_result = [] 
    uiquid = []
    epsclf = None
    dens_ctr = None
    gap = 1
    RMSE_result = list()
    BTs = timeit.default_timer()
    model = onoff.Model(0.999, cm*1000, aid, appro)
    adv = advertiser.ADer(len(dates), cm*1000, bu, gaphour, appro, aid)
    
    print ("*****Start Training*****")
    pretrain_days = 3 
    flowpretrain_days = 3 
    
    new_day = -1
    flow_latency = 0
    predict_latency = 0
    for i in range(0, len(dates)):
        if i < pretrain_days:
            label = True
        else:
            label = False
            break
        DTs = timeit.default_timer()
        
        if i == 0:
            
            daily_raw = model.get_data(dates[i][0], dates[i][1])
            daily_train, daily_outbid, daily_label, daily_winp, daily_bid, ctrlist, click, f, epsout, epswp = model.traindata_prepare_hash(daily_raw, model.feature_size, allday = True)
            daily_model = model.train_model_hash(daily_train, daily_label, SGD = False)
            epsclf =  SGDRegressor(penalty='l2', alpha = 0.01, fit_intercept=True, n_iter_no_change  = 1, shuffle = False, warm_start = True, learning_rate = 'constant', eta0 = 0.000000005)
            epsclf.fit(epsout, epswp)
            
            del daily_raw
            del daily_outbid
            del daily_bid
            del daily_train
            del f
            daily_raw = []
            daily_winp = []
            daily_outbid = []
            daily_bid = []
            daily_train = []
            daily_label = []
            f = []
            print ("First day completed!", type(daily_model))
            continue
        else:
            if i != 1 and label:
                daily_model = model.train_model_hash(pre_day.tocsr()[1:,].tocoo(), np.array(pre_label))
            for h in range(0, len(hours), gaphour):
                if label == False:
                    if isinstance(adv.stat, list):
                        adv.stat = np.mean(adv.stat)
                    print ('eachcosts ', adv.costs, 'allbudget ', adv.budget)                
                print (hours[h])
                
                if new_day != i and i != pretrain_days and label == False:
                    new_day = i
                    
                hourly_temp_label = list()
                hourly_raw = model.get_data(dates[i][0], dates[i][1], aid, hours[h], 'hourly', gaphour)
                hourly_train, hourly_outbid, hourly_label, hourly_winp, hourly_bid, ctrlist, click, f = model.traindata_prepare_hash(hourly_raw, model.feature_size)
       
                del hourly_raw
                
                if isinstance(hourly_train, int):
                    print ('no data')
                    continue
                if hourly_train.shape[0]<= 1:
                    continue
               

                start_flow = 0
                end_flow = 0

                if label == True and i >= flowpretrain_days:
                    
                    adv.requests = hourly_train.shape[0]
                elif label == False and i >= flowpretrain_days:
                    adv.requests = 0
                    etime = timeit.default_timer()
                    for ph in range(h, h+gaphour):
                        
                        start_flow = timeit.default_timer()
                        
                        end_flow = timeit.default_timer()
                        flow_latency += (end_flow - start_flow)
                        
                        hourground = model.get_data(dates[i][0], dates[i][1], aid, hours[ph], 'hourly').count()
                       
                    print ('flow time: ', timeit.default_timer()-etime)
                else:
                    adv.requests = hourly_train.shape[0]
                
      

                request_ground = hourly_train.shape[0]
                
                del hourly_train
                hourly_train = []
                
                if (h == 0 or new_day != i) and label:
                    pre_day = csr_matrix((1,model.feature_size+1))
                    pre_label = list()
                    if i != 1:
                        del hourly_model
                    hourly_model = daily_model
                    del daily_model
                if adv.ad_performance != "performance":
                    start_predict = timeit.default_timer()
                    prd_bid, prob, epsclf = model.given_recommend_price_wr(hourly_outbid, model.feature_size, hourly_model.coef_[0], hourly_model.intercept_[0], hourly_winp, adv, label, click, ctrlist, dens_ctr, th, epsclf, ct, record = False, adid = aid)
                    end_predict = timeit.default_timer() - start_predict
                    predict_latency += end_predict
                    print ("Time Slot Predict Latency: ", end_predict, "Requests: ", len(prd_bid), "Every Request Latency, ", end_predict/len(prd_bid)) 
                    
                else:
                    prd_bid, prob = model.given_recommend_price_wr_performance(hourly_outbid, model.feature_size, hourly_model.coef_[0], hourly_model.intercept_[0], hourly_winp, adv, label, ctrlist, click)
                model.cpm_eva(prd_bid, hourly_winp[:len(prd_bid)])

                hourly_temp = hstack([hourly_outbid[:len(prd_bid),], csr_matrix(np.reshape(prd_bid, (-1, 1)))])
                
                if label:
                    pre_day = model.combine_csrvstack(pre_day, hourly_temp.tocsr())
               
                for la in range(0, len(prd_bid)):
                    
                    if float(prd_bid[la]) > hourly_winp[la]:
                        pre_label.append(True)
                        hourly_temp_label.append(True)
                        if label == False:
                            uiquid.append(f[la])
                    else:
                        pre_label.append(False)    
                        hourly_temp_label.append(False)
                if new_day != i and label:
                    new_hourly_model = model.train_model_hash(hourly_temp, np.array(hourly_temp_label), hourly_model, True)
                    del hourly_model
                    hourly_model = new_hourly_model
                    del new_hourly_model
                    new_day = i
                elif label:
                    new_hourly_model = model.train_model_hash(hourly_temp, np.array(hourly_temp_label), hourly_model, False)
                    del hourly_model
                    hourly_model = new_hourly_model
                    del new_hourly_model
                del hourly_outbid
                del hourly_temp
                 
        print ("Each Day Training %s" %(timeit.default_timer()-DTs))
    print ("Approx. Type: ", appro, "=====Finish Training Data: %s=====" %(timeit.default_timer()-BTs))

   
    model.client.close()
    print(model.clusters)
    return model, hourly_model, f
    
if __name__ == '__main__':

    cpm = [300.]
    budget = [500000.]
    gaphours = [1]
     
    ctrprolist = [0.]
    for aid in ['1458', '3386']:
        if aid == '3386' and approxi == 3:
            ctrprolist = [0]#[0.5]
        if aid == '1458' and approxi == 3:
            ctrprolist = [0]#[10.]

        
        for th in [1.5]:
            for gh in gaphours:
               
                for b in budget:
                    for cp in cpm:
                        for c in ctrprolist:
                            mm = main(0, cp, b, aid, gh, th, appro = str(approxi), ct = c)
                            mm[0].client = 0
                            with open('{0}.model'.format(aid), 'wb') as mo:   
                                pickle.dump(mm, mo)

