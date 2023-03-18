import pickle, os, sys, timeit, random, math

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
import onoff, advertiser, Traffic, Flow_Pred
import warnings                                                                                                   
warnings.filterwarnings("ignore", category=DeprecationWarning)
#fp=open('memory_profiler3h.log','w')
dates = [("10","01"), ("10","02"), ("10","03"), ("10","04"), ("10","05"), ("10","06"), ("10","07")]
#dates = [("06","06"), ("06","07"), ("06","08"), ("06","09"), ("06","10"), ("06","11"), ("06","12")]#, ("10","19"), ("10","20"), ("10","21"), ("10","22"), ("10","23"), ("10","24"), ("10","25"), ("10","26"), ("10","27")]
#dates = ["19", "20", "21", "22", "23", "24", "25", "26", "27"]
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

#if __name__ == '__main__':
#@profile(stream=fp)
#sys.stdout=os.fdopen(sys.stdout.fileno(), 'w', 0)
globalappro = 3 
def main(noise=None, cm=None, bu=None, firsttr = True, hourly_model = None, gaphour = 1, dens_ctr = None, th = .999, appro = '1', epsclf = None, ctrpro = 0., aid = None, avgclictr = None, ctrtrain = False):  
    #gc.set_threshold(100, 10, 10) 
    var_result = []
    #RMSE_result = []
    #Acc_result = []
    slope_result = [] 
    uiquid = []
    #model = Model()
    gap = 1
    RMSE_result = list()
    BTs = timeit.default_timer()
    model = onoff.Model(0.999, cm*1000, appro, aid, ctrtrain, firsttr)
    adv = advertiser.ADer(len(dates), cm*1000, bu, gaphour, appro, aid)
    #traffic = Traffic.Traffic()
    #print "*****Start Training*****"
    pretrain_days = 3 
    flowpretrain_days = 3 
    #traffic = Flow_Pred.Traffic(flowpretrain_days, aid)
    new_day = -1
    record = False
    flow_latency = 0
    predict_latency = 0
    if firsttr:
        ckk = 0
    else:
        ckk = 1
    for i in range(0+ckk*pretrain_days, len(dates)):
        if i < pretrain_days and firsttr == True:
            label = True
        else:
            label = False
            if firsttr == False and ctrtrain == True:
                break
        DTs = timeit.default_timer()
        #print dates[i], label
        #pre_day = csr_matrix((1,model.feature_size+1))
        #pre_label = list()
        if i == 0 and hourly_model == None:
            
            daily_raw = model.get_data(dates[i][0], dates[i][1])
            daily_train, daily_outbid, daily_label, daily_winp, daily_bid, ctrlist, click, f, epsout, epswp = model.traindata_prepare_hash(daily_raw, model.feature_size, allday = True)
            daily_model = model.train_model_hash(daily_train, daily_label, SGD = False)
            epsclf =  SGDRegressor(penalty='l2', alpha = 0.01, fit_intercept=True, shuffle = False, warm_start = True, learning_rate = 'constant', eta0 = 0.0000005)
            epsclf.fit(epsout, epswp)
            ctr = [math.log(i) for i in ctrlist]
            clictr = [ctrlist[j] for j in range(0, len(click)) if click[j]]
            if avgclictr == None:
                avgclictr = sorted(clictr)[min(len(clictr)-1, int(len(clictr)*ctrpro))]
            #dens_ctr = norm.fit(ctr)
            dens_ctr = None
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
            #print "First day completed!", type(daily_model)
            continue
        else:
            if i != 1 and label and firsttr == True:
                daily_model = model.train_model_hash(pre_day.tocsr()[1:,].tocoo(), np.array(pre_label))
            if firsttr == True and label == False:
                return model, hourly_model, FeatureHasher(pow(2, 7))#hourly_model, dens_ctr, epsclf, avgclictr
            for h in range(0, len(hours), gaphour):
                if label == False:
                    print ('eachcosts ', adv.costs, 'allbudget ', adv.budget)
                    #if h == 0 and i == 4:
                    #    sys.exit()         
                #print hours[h]
                if new_day != i and i != pretrain_days and label == False:
                    new_day = i
                    #traffic.daily_train()
                #if i == pretrain_days and h == 12 and gaphour == 1:
                #    record = True
                #else:
                #    record = False
                    #if  i == pretrain_days and h == 6 and gaphour == 6:
                    #    sys.exit()
                hourly_temp_label = list()
                hourly_raw = model.get_data(dates[i][0], dates[i][1], hours[h], 'hourly', gaphour, aid)        
                hourly_train, hourly_outbid, hourly_label, hourly_winp, hourly_bid, ctrlist, click, f = model.traindata_prepare_hash(hourly_raw, model.feature_size)
                #hourly_raw = []
                del hourly_raw
                # key to reduce memory, because origin way causes sparse matrix to be used as normal matrix
                if isinstance(hourly_train, int):
                    continue
                if hourly_train.shape[0]<= 1:
                    continue
                #adv.requests = int(hourly_train.shape[0] + random.uniform(-1*noise, noise)* hourly_train.shape[0])
                #adv.requests = hourly_train.shape[0]
                if label == True and i >= flowpretrain_days:
                    #adv.requests = int(traffic.predict(int(hours[h])))
                    adv.requests = hourly_train.shape[0]
                elif label == False and i >= flowpretrain_days:
                    adv.requests = 0.
                    start_flow = 0
                    end_flow = 0
                    for ph in range(h, h+gaphour):
                        start_flow = timeit.default_timer() 
                        #temp_flow = int(traffic.predict(int(hours[ph])))
                        end_flow = timeit.default_timer()
                        flow_latency += (end_flow - start_flow)
                        adv.requests += temp_flow
                        hourground = model.get_data(dates[i][0], dates[i][1], hours[ph], 'hourly', aid = aid).count()
                        #traffic.hourly_update(hourground, int(hours[ph]))
                else:
                    adv.requests = hourly_train.shape[0]
                #adv.requests = hourly_train.shape[0]
                request_ground = hourly_train.shape[0]
                if label == False:
                    print ("request number: ", adv.requests, hourly_train.shape[0], float(adv.requests-hourly_train.shape[0])/hourly_train.shape[0], "temp_flow_latency: ", end_flow - start_flow)
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
                    prd_bid, prob, epsclf = model.given_recommend_price_wr(hourly_outbid, model.feature_size, hourly_model.coef_[0], hourly_model.intercept_[0], hourly_winp, adv, label, click, ctrlist, dens_ctr, th, epsclf, ctrpro = ctrpro, record = record)
                    end_predict = timeit.default_timer()
                    temp_predict_latency = end_predict - start_predict
                    predict_latency +=  temp_predict_latency
                    print ("Time Slot Predict Latency: ", temp_predict_latency, "Requests: ", len(prd_bid), "Every Request Latency, ", temp_predict_latency/len(prd_bid))
                    #prd_bid, prob, epsclf = model.given_recommend_price_wr(hourly_outbid, model.feature_size, hourly_model.coef_[0], hourly_model.intercept_[0], hourly_winp, adv, label, click, ctrlist, dens_ctr, th, epsclf, ctrpro = avgclictr, record = record)
                    #if record:
                    #    sys.exit()
                else:
                    prd_bid, prob = model.given_recommend_price_wr_performance(hourly_outbid, model.feature_size, hourly_model.coef_[0], hourly_model.intercept_[0], hourly_winp, adv, label, ctrlist, click)
                #model.cpm_eva(prd_bid, hourly_winp[:len(prd_bid)])

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
                if firsttr:
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
                    #new_hourly_model = model.train_model_hash(hourly_temp, np.array(hourly_temp_label), hourly_model, True)
                    #del hourly_model
                    #hourly_model = new_hourly_model
                    #del new_hourly_model
                    new_day = i
                #elif label:
                    #new_hourly_model = model.train_model_hash(hourly_temp, np.array(hourly_temp_label), hourly_model, False)
                    #del hourly_model
                    #hourly_model = new_hourly_model
                    #del new_hourly_model
                del hourly_outbid
                del hourly_temp
                 
        #print "Each Day Training %s" %(timeit.default_timer()-DTs)
    print ("Approx. Type, ", appro, "=====Finish Training Data: %s=====" %(timeit.default_timer()-BTs))
    if float(adv.clicks) != 0 and ctrtrain:
        print ('Adid, ', aid, 'total budget, ', bu, "total cost, ", adv.costs, "total impressions, ", adv.imps, "total clicks, ", adv.clicks, "Upprice", cm, 'hourgap, ', gaphour, "CTR Promote, ", ctrpro, "CPM, ", adv.costs*1000/float(adv.imps), "CPC, ", adv.costs/float(adv.clicks), "H(), ", 2./(1/(adv.costs*1000./adv.imps)+1/(1.*adv.costs/adv.clicks)), "flow latency, ", flow_latency, "predict latency, ", predict_latency)
    #elif ctrtrain:
        #print ('Adid, ', aid, 'total budget, ', bu, "total cost, ", adv.costs, "total impressions, ", adv.imps, "total clicks, ", adv.clicks, "Upprice", cm, 'hourgap, ', gaphour, "CTR Promote, ", ctrpro, "CPM, ", adv.costs/float(adv.imps), "CPC, ", 0, "flow latency, ", flow_latency, "predict latency, ", predict_latency)

    model.client.close()
    if ctrtrain:
        print(model.clusters)
    return model, hourly_model, FeatureHasher(pow(2, 7)) 
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
    gaphours = [1]#, 2, 3, 4, 6]
    #ctrprolist = [0.8, 0.85, 0.9, 0.95, 1.]#[0., 0.5, 1.0, 2.0, 5.0, 10.]#np.linspace(0, 1, 11)
    ctrprolist = [0]#[1.]#[0., 0.5, 1.0, 2.0, 5.0, 10.]
    for adid in ['215']:
        for gh in gaphours:
            #if gh == 1 or gh == 2:
            #    ctrprolist = [0., 0.5, 1.0, 2.0, 5.0, 10.]
            #else:
            #    ctrprolist = [0.]
            for ct in ctrprolist:
                mm = main(0, 601., 5000000, gaphour = gh, appro = str(globalappro), ctrpro = ct, ctrtrain = True, aid = adid)
                cpm = [601.]#44.]#, 130.]#, 1.5, 10]
                budget = [50000000.]#[500000., 1000000., 1500000.]#, 5000000.]
                for th in [1.5]:#[.8,.7,.6]:
                    for b in budget[0:1]:
                        for cp in cpm[0:1]:
                        #for ct in ctrprolist:
                            #mm = main(0, cp, b, False, model, gh, dens_ctr, th, appro = str(globalappro), epsclf = epsclf, aid = adid, ctrpro = ct, ctrtrain = True)#, avgclictr = avgclictr)#ct)
                            mm[0].client = 0
                            with open('{0}.model'.format(adid), 'wb') as mo:
                                pickle.dump(mm, mo)
    #for b in budget:
        #main(0, 50000000, b, False, model, 6, dens_ctr) 
    #    main(0, 500, b)
    #main(0, 500, 1000000.)
