import pickle, os, sys, timeit, random, math
from copy import deepcopy

sys.path.append('/home/mpc/RTB/schema/')

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
#from sklearn.cross_validation import ShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
import matplotlib.pyplot as plt
#from memory_profiler import profile
#import objgraph
import onoff, advertiser#, Traffic, Flow_Pred
import warnings                                                                                                   
warnings.filterwarnings("ignore", category=DeprecationWarning)
#fp=open('memory_profiler3h.log','w')
dates = [("06","06"), ("06","07"), ("06","08"), ("06","09"), ("06","10"), ("06","11"), ("06","12")]#,
#dates = [("10","19"), ("10","20"), ("10","21"), ("10","22"), ("10","23"), ("10","24"), ("10","25"), ("10","26"), ("10","27")]
#dates = ["19", "20", "21", "22", "23", "24", "25", "26", "27"]
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

#if __name__ == '__main__':
#@profile(stream=fp)
#sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
#def trainspeed(med, medbu, end, endbu):

approxi = 3 
    
#def speedforece():
def main(noise, cm, bu, aid= None, gaphour = 1, th = .999, appro = '1', ct = 0.): 
    #gc.set_threshold(100, 10, 10) 
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
    #traffic = Flow_Pred.Traffic(flowpretrain_days, aid)
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
        print (dates[i], label)
        #pre_day = csr_matrix((1,model.feature_size+1))
        #pre_label = list()
        if i == 0:
            
            daily_raw = model.get_data(dates[i][0], dates[i][1])
            daily_train, daily_outbid, daily_label, daily_winp, daily_bid, ctrlist, click, f, epsout, epswp = model.traindata_prepare_hash(daily_raw, model.feature_size, allday = True)
            daily_model = model.train_model_hash(daily_train, daily_label, SGD = False)
            epsclf =  SGDRegressor(penalty='l2', alpha = 0.01, fit_intercept=True, n_iter_no_change  = 1, shuffle = False, warm_start = True, learning_rate = 'constant', eta0 = 0.000000005)
            epsclf.fit(epsout, epswp)
            #ctr = [math.log(i) for i in ctrlist]
            #dens_ctr = norm.fit(ctr)
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
                #if label == False and h == 1*gaphour:
                #    sys.exit()
                if new_day != i and i != pretrain_days and label == False:
                    new_day = i
                    #traffic.daily_train()
                #if i == pretrain_days and ((h == 9 and aid == '3386') or (h == 11 and aid == '1458')) and gaphour == 1:
                #    record = True
                #else:
                #    record = False
                hourly_temp_label = list()
                hourly_raw = model.get_data(dates[i][0], dates[i][1], aid, hours[h], 'hourly', gaphour)
                hourly_train, hourly_outbid, hourly_label, hourly_winp, hourly_bid, ctrlist, click, f = model.traindata_prepare_hash(hourly_raw, model.feature_size)
                #hourly_raw = []
                del hourly_raw
                # key to reduce memory, because origin way causes sparse matrix to be used as normal matrix
                if isinstance(hourly_train, int):
                    print ('no data')
                    continue
                if hourly_train.shape[0]<= 1:
                    continue
                #adv.requests = int(hourly_train.shape[0] + random.uniform(-1*noise, noise)* hourly_train.shape[0])
                #adv.requests = hourly_train.shape[0]

                start_flow = 0
                end_flow = 0

                if label == True and i >= flowpretrain_days:
                    #adv.requests = int(traffic.predict(int(hours[h])))
                    adv.requests = hourly_train.shape[0]
                elif label == False and i >= flowpretrain_days:
                    adv.requests = 0
                    etime = timeit.default_timer()
                    for ph in range(h, h+gaphour):
                        #print ph, hours[ph], int(traffic.predict(int(hours[ph])))
                        start_flow = timeit.default_timer()
                        #predicted_flow = int(traffic.predict(int(hours[ph])))
                        end_flow = timeit.default_timer()
                        flow_latency += (end_flow - start_flow)
                        #adv.requests += predicted_flow#int(traffic.predict(int(hours[ph])))
                        hourground = model.get_data(dates[i][0], dates[i][1], aid, hours[ph], 'hourly').count()
                        #traffic.hourly_update(hourground, int(hours[ph]))
                    print ('flow time: ', timeit.default_timer()-etime)
                else:
                    adv.requests = hourly_train.shape[0]
                
                #adv.requests = hourly_train.shape[0]

                request_ground = hourly_train.shape[0]
                #print "request number: ", adv.requests, hourly_train.shape[0], float(adv.requests-hourly_train.shape[0])/hourly_train.shape[0], "temp flow latency, ", end_flow - start_flow
                #print "request number: ", adv.requests, 'ground_truth: ', request_ground 
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
                #if record:
                #    sys.exit()
                hourly_temp = hstack([hourly_outbid[:len(prd_bid),], csr_matrix(np.reshape(prd_bid, (-1, 1)))])
                #pre_day = vstack([pre_day, hourly_temp])
                if label:
                    pre_day = model.combine_csrvstack(pre_day, hourly_temp.tocsr())
                #else:
                #    traffic.hourly_update(request_ground, int(hours[h]))
                #del pre_day
                #pre_day = pre_day1
                #del pre_day1
                #rmse, slope = model.ratio_eva_hash(hourly_outbid, model.feature_size, hourly_model.coef_[0], hourly_model.intercept_[0], hourly_winp)
                #RMSE_result.append(str(dates[i])+" "+str(hours[h])+" "+str(rmse))
                #slope_result.append(str(dates[i])+" "+str(hours[h])+" "+str(slope))
                for la in range(0, len(prd_bid)):#len(hourly_winp)):
                    
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

    #print ('Adid, ', aid, 'total budget, ', bu, "total cost, ", adv.costs, "total impressions, ", adv.imps, "total clicks, ", adv.clicks, "H(), ", 2./(1/(adv.costs*1000./adv.imps)+1/(adv.costs/float(adv.clicks))),"Upprice", cm, "CTR Pro., ", ct, "gap hour, ", gaphour, "CPM, ", adv.costs*1000./adv.imps, "flow latency,  ", flow_latency, "predict_latency, ", predict_latency)
    model.client.close()
    print(model.clusters)
    return model, hourly_model, f
    #print "-----------------------------------slope-----------------------------------------"
    #for i in slope_result:
    #    print i
    #print "other model---------------+++++++++++++++"
    #os.system("nohup ipython onlineoffline_sgd.py &")
if __name__ == '__main__':
    #sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
    #noise = np.linspace(0.4, 1, 7)
    #for i in noise:
    #    main(i)
    cpm = [300.]#55., 76.]
    budget = [500000.]#, 1000000., 1500000.]
    gaphours = [1]#, 2, 3, 4, 6]
     
    ctrprolist = [0.]#, 0.5, 1.0, 5.0, 10.]#np.linspace(0., 1., 10)
    for aid in ['1458']:#, '3358', '1458']:
        if aid == '3386' and approxi == 3:
            ctrprolist = [0]#[0.5]
        if aid == '1458' and approxi == 3:
            ctrprolist = [0]#[10.]
        if aid == '3427' and approxi == 3:
            ctrprolist = [0]
        #if aid == '3386' and approxi == 4:
        #    ctrprolist = [0.]           
        #if aid == '1458' and approxi == 4:
        #    ctrprolist = [5.]
        
        for th in [1.5]:#[.999, .995, .99, .95, .90]:
            for gh in gaphours:
                #if gh == 1 or gh == 2:
                #    ctrprolist = [0., 0.5, 1.0, 5.0, 10.]
                #else:
                #    ctrprolist = [0.]
                for b in budget:
                    for cp in cpm:
                        for c in ctrprolist:
                            mm = main(0, cp, b, aid, gh, th, appro = str(approxi), ct = c)
                            mm[0].client = 0
                            with open('{0}.model'.format(aid), 'wb') as mo:   
                                pickle.dump(mm, mo)
    #for b in budget:
    #    main(0, 500, b, None, 6)
    #main(0, 500, 1000000.)
