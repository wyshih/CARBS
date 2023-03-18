import pickle, os, sys, timeit, random, math

sys.path.append('/home/mpc/RTB/schema/')

import itertools, gc
import pandas as pd
from pymongo import MongoClient
import datetime
from scipy.stats import norm
import numpy as np
import scipy
from scipy.stats import lognorm, linregress
from scipy.sparse import vstack, hstack, csr_matrix, coo_matrix
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDRegressor, Ridge
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.preprocessing import scale, normalize
#from sklearn.cross_validation import ShuffleSplit
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import matplotlib, copy
import matplotlib.pyplot as plt
import advertiser
#from memory_profiler import profile
#fp=open('memory_profiler_h.log','w')
dates = [("06","06"), ("06","07"), ("06","08"), ("06","09"), ("06","10"), ("06","11"), ("06","12"), ("10","19"), ("10","20"), ("10","21"), ("10","22"), ("10","23"), ("10","24"), ("10","25"), ("10","26"), ("10","27")]
#dates = ["19", "20", "21", "22", "23", "24", "25", "26", "27"]
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
class Model:
    
    
    def __init__(self, uwr, cpm, aid, appro = None):
        self.dates = dates
        self.hours = hours
        self.bid_floor = 1.
        self.bid_upper = 301.
        self.feature_size = pow(2, 9)
        self.winrate_low = 0.001
        self.winrate_up = uwr#0.999
        self.cpm = cpm
        self.appro = appro
        self.clusters = []
        self.ctrf = None
        self.client = MongoClient('Enter your data source')
        try:
            with open('{0}.ctrmodel'.format(aid), 'rb') as a:
                self.ctrf, self.ctrmodel = pickle.load(a)
            self.train_ctrmodel(aid, 8, self.ctrmodel, self.ctrf)
        except:
            self.ctrf, self.ctrmodel = self.train_ctrmodel(aid, 8)
    def loaddata(self, begintime, endtime, aid = None, hour=None, obj = 'daily', gaphour = 1):
        db = self.client.iPinYou_allinone
        if obj == 'daily':
            time1 = begintime# datetime.datetime.strptime('2013-'+month+'-'+date, '%Y-%m-%d')
            time2 = endtime#time1+datetime.timedelta(days=1)
        else:
            time1 = begintime#datetime.datetime.strptime('2013-'+month+'-'+date+' '+hour, '%Y-%m-%d %H')
            time2 = time1 + datetime.timedelta(hours=gaphour)
        if aid != None:
            cursor = db.ipinyou.find({'timestamp':{"$gte": time1, '$lt': time2}, "bidding_price": {"$gt": 0.}, "paying_price": {"$gt":0.}, 'advertiser_id':aid})
        else:
            cursor = db.ipinyou.find({'timestamp':{"$gte": time1, '$lt': time2}, "bidding_price": {"$gt": 0.}, "paying_price": {"$gt":0.}})
        return cursor

    def get_data(self, month, date, aid = None, hour=None, obj = 'daily', gaphour = 1): 
        db = self.client.iPinYou_allinone
        if obj == 'daily':
            time1 = datetime.datetime.strptime('2013-'+month+'-'+date, '%Y-%m-%d')
            #time2 = datetime.datetime.strptime('2013-'+month+'-'+str(int(date)+1), '%Y-%m-%d')
            time2 = time1+datetime.timedelta(days=1)
        else:
            time1 = datetime.datetime.strptime('2013-'+month+'-'+date+' '+hour, '%Y-%m-%d %H')
            time2 = time1 + datetime.timedelta(hours=gaphour)
            #if hour != '23':
            #    time2 = datetime.datetime.strptime('2013-'+month+'-'+date+' '+str(int(hour)+1), '%Y-%m-%d %H')
            #else:
            #    time2 = datetime.datetime.strptime('2013-'+month+'-'+str(int(date)+1)+' 00', '%Y-%m-%d %H')
        if aid != None:
            cursor = db.ipinyou.find({'timestamp':{"$gte": time1, '$lt': time2}, "bidding_price": {"$gt": 0.}, "paying_price": {"$gt":0.}, 'advertiser_id':aid})
        else:
            cursor = db.ipinyou.find({'timestamp':{"$gte": time1, '$lt': time2}, "bidding_price": {"$gt": 0.}, "paying_price": {"$gt":0.}})
        #print cursor.count()
        return cursor
    def train_ctrmodel(self, aid, k, clf = None, f = None):
        trainbegintime = datetime.datetime.strptime('2013-06-06', '%Y-%m-%d')
        trainendtime = trainbegintime+datetime.timedelta(days=2)

        validatendtime = trainendtime+datetime.timedelta(days=1)
        testendtime = validatendtime+datetime.timedelta(days=4)
        cur = self.loaddata(trainbegintime, validatendtime, aid)
        if clf == None:
            _, train_outbid, label, wps, _, _, click, f = self.traindata_prepare_hash(cur, pow(2, 20), ctrtrain = True)
        else:
            _, train_outbid, label, wps, _, _, click, _ = self.traindata_prepare_hash(cur, pow(2, 20), ctrtrain = True, f = f)

        print(len(click), sum(click), train_outbid.shape)
        if clf == None:
            ctrclf = LogisticRegression(penalty='l2', dual=False)
            para = {'C': [1, 1e-2]}
            clf = GridSearchCV(ctrclf, para, cv=3, n_jobs = -1, scoring='roc_auc')
            clf.fit(train_outbid, click)
        print ("AD: ", aid, "=================")
        trainctr = [i[1] for i in clf.best_estimator_.predict_proba(train_outbid)]
        begin = datetime.datetime.now()
        ra = list(np.geomspace(min(trainctr), max(trainctr), num=20, endpoint=False))
        totime = datetime.datetime.now()-begin
        print('lbinning time', 1000*(totime.total_seconds()), ra)
        '''ctrpd = pd.DataFrame(trainctr, columns=['ctr'])
        begin = datetime.datetime.now()
        ra = pd.cut(ctrpd.loc[:,'ctr'], 20, retbins=True)[1]
        totime = datetime.datetime.now()-begin
        print('cbinning time', 1000*(totime.total_seconds()), ra)
        begin = datetime.datetime.now()
        ra = pd.qcut(ctrpd.loc[:,'ctr'], 20, retbins=True)[1]
        totime = datetime.datetime.now()-begin
        print('qbinning time', 1000*(totime.total_seconds()), ra)
        ra = list(ra)#(copy.deepcopy(mp))
        '''
        x = {}
        y = {}
        z = {}
        avgctr = {}
        countnum = {}
        wpk = {}
        numk = {}
        ra.reverse()
        ra.append(0.)
        for i in ra:
            x.setdefault(i, [0, 0.0000001])
            y.setdefault(i, 0.)
            z.setdefault(i, [])
            avgctr.setdefault(i, [])
            countnum.setdefault(i, 0)
        for i in range(len(trainctr)):
            for j in ra:
                if trainctr[i] >= j:
                    countnum[j]+= 1.
        i = 0
        print(ra, countnum)
        while(1):
            if countnum[ra[i]] < 50:
                ra.remove(ra[i])
                i -= 1
                if i < 0:
                    i = 0
            else:
                i += 1
            if i == len(ra)-1:
                break
        print(ra)
        for i in range(len(trainctr)):
            for j in ra:
                if trainctr[i] >= j:
                    avgctr[j].append(trainctr[i])
                    if click[i]:
                        x[j][0] += 1
                    x[j][1] += 1.
                    if label[i]:
                        z[j].append(wps[i])
                    break
        for i in x.keys():
            y[i] = x[i][0]/x[i][1]
        print("avg real: ", [(np.mean(avgctr[i]), y[i]) for i in ra])
        mm = sum([x[i][1] for i in x.keys()])*1.0
        nx = []
        for i in ra:
            #nx.append((i, y[i]/(x[i][1]/mm)))
            nx.append((i, y[i]))#, x[i][1]/mm))
        for mm in range(3, 11):
            begin = datetime.datetime.now()
            kmeans = KMeans(n_clusters=mm).fit(nx)
            totime = datetime.datetime.now()-begin
            print("SSE for ", mm, " : ", kmeans.inertia_)
            print('cluster time', 1000*(totime.total_seconds()))
        kmeans = KMeans(n_clusters=k).fit(nx)
        for im in range(k):
            numk.setdefault(im, 0.)
            wpk.setdefault(im, [])
            temp = 0.
            tempwp = []
            for ii in range(len(ra)):
                if list(kmeans.labels_)[ii] == im:
                    temp += (x[ra[ii]][1]-0.0000001)
                    tempwp.extend(z[ra[ii]])
            numk[im] = temp/len(trainctr)
            wpk[im].extend([max(tempwp), sorted(tempwp)[int(len(tempwp)*0.8)], sorted(tempwp)[int(len(tempwp)*0.7)], sorted(tempwp)[int(len(tempwp)*0.6)], np.mean(tempwp), scipy.stats.mstats.gmean(tempwp), sum(tempwp), len(tempwp)])
        self.clusters = [ra, list(kmeans.labels_), list(kmeans.cluster_centers_), numk, wpk]
        print (clf.best_params_ , ": AUC", clf.best_score_ )

        tcursor = self.loaddata(trainendtime, validatendtime, aid)
        _, test_outbid, _, _, _, _, tclick, _ = self.traindata_prepare_hash(tcursor, pow(2, 20), ctrtrain = True, f = f)
        tvalctr = [i[1] for i in clf.predict_proba(test_outbid)]
        print ("Validation AUC", metrics.roc_auc_score(tclick, tvalctr))

        tcursor = self.loaddata(validatendtime, testendtime, aid)
        _, test_outbid, _, _, _, _, tclick, _ = self.traindata_prepare_hash(tcursor, pow(2, 20), ctrtrain = True, f = f)
        tvalctr = [i[1] for i in clf.predict_proba(test_outbid)]
        print ("AUC", metrics.roc_auc_score(tclick, tvalctr))
        if self.ctrf == None:
            with open('{0}.ctrmodel'.format(aid), 'wb') as abd:
                pickle.dump([f, clf], abd)
            with open('ctrmodel.log', 'a+') as dd:
                dd.write('AD: {0}-------\n'.format(aid))
                dd.write(str(y)+'\n')
                dd.write(str(nx)+'\n')
                dd.write(str(wpk)+'\n')
            return f, clf.best_estimator_
        print('exist ctrmodel!')
    
    def data_prepare(self, each, feature_size, ctrtrain = False):
        x = dict()
        x.setdefault('IP'+"="+str(each['ip']), 1)
        x.setdefault('Region'+"="+str(each['region']), 1)
        x.setdefault('CityID'+"="+str(each['city']), 1)
        x.setdefault('AdExchange'+"="+str(each['ad_exchange']), 1)
        x.setdefault('URL'+"="+str(each['url']), 1)
        x.setdefault('Domain'+"="+str(each['domain']), 1)
        x.setdefault('AdSlotID'+"="+str(each['ad_slot_id']), 1)
        #x.setdefault('AdSlotWidth'+"="+str(each['ad_slot_width']), 1)
        #x.setdefault('AdSlotHeight'+"="+str(each['ad_slot_height']), 1)
        if ctrtrain == False:
            x.setdefault('AdSlotWidth', int(each['ad_slot_width']))
        x.setdefault('AdSlotheight', int(each['ad_slot_height']))
        x.setdefault('advertiser_id'+'='+str(each['advertiser_id']), 1)
        x.setdefault('AdSlotVisibility'+"="+str(each['ad_slot_visibility']), 1)
        x.setdefault('AdSlotFormat'+"="+str(each['ad_slot_format']), 1)
        x.setdefault('hour'+"="+str(each['timestamp'].hour), 1)
        #x.setdefault('ctr', float(each['lg_ctr']))
        tags = each['user_tags'].strip().split(",")
        for tag in tags:
            x.setdefault('UserTags'+"="+str(tag), 1)
        if ctrtrain == False:
           x.setdefault('ctr', self.ctrmodel.predict_proba(self.ctrf.transform([x]))[0][1])
        
           f = FeatureHasher(feature_size)
        else:
           f = self.ctrf
        train_x = f.fit_transform([x])
        return train_x#, float(each['paying_price'])
    def traindata_prepare_hash(self, train, feature_size, f = None, allday = False, ctrtrain = False):
        userid = list()
        train_data = list()
        train_label = list()
        train_win_p = list()
        train_bid = list()
        ctr = list()
        click = list()
        epsmatrix = list()
        epsbid = list()
        epswp = list()
        count = 0
        for each in train:
            userid.append(str(each['bid_id']))
            x = dict()
            x.setdefault('IP'+"="+str(each['ip']), 1)
            x.setdefault('Region'+"="+str(each['region']), 1)
            x.setdefault('CityID'+"="+str(each['city']), 1)
            x.setdefault('AdExchange'+"="+str(each['ad_exchange']), 1)
            x.setdefault('URL'+"="+str(each['url']), 1)
            x.setdefault('Domain'+"="+str(each['domain']), 1)
            x.setdefault('AdSlotID'+"="+str(each['ad_slot_id']), 1)
            x.setdefault('advertiser_id'+'='+str(each['advertiser_id']), 1)
            #x.setdefault('AdSlotWidth'+"="+str(each['ad_slot_width']), 1)
            #x.setdefault('AdSlotHeight'+"="+str(each['ad_slot_height']), 1)
            if ctrtrain == False:
                x.setdefault('AdSlotWidth', int(each['ad_slot_width']))
            x.setdefault('AdSlotheight', int(each['ad_slot_height']))
            x.setdefault('AdSlotVisibility'+"="+str(each['ad_slot_visibility']), 1)
            x.setdefault('AdSlotFormat'+"="+str(each['ad_slot_format']), 1)
            x.setdefault('hour'+"="+str(each['timestamp'].hour), 1)
            tags = each['user_tags'].strip().split(",")
            for tag in tags:
                x.setdefault('UserTags'+"="+str(tag), 1)
            if ctrtrain == False:
                x.setdefault('ctr', self.ctrmodel.predict_proba(self.ctrf.transform([x]))[0][1])
                '''try:
                    print(x['ctr'], each['lg_ctr'], each['click'])
                except:
                    pass
                '''
            '''
            #x.setdefault('BidID', each['BidID'])
            x.setdefault(str('IP'+"="+str(each.bidRequest.IP)), 1)
            x.setdefault('Region'+"="+str(each.bidRequest.Region), 1)
            x.setdefault('CityID'+"="+str(each.bidRequest.CityID), 1)
            x.setdefault('AdExchange'+"="+str(each.bidRequest.AdExchange), 1)
            x.setdefault('URL'+"="+str(each.bidRequest.URL), 1)
            x.setdefault('Domain'+"="+str(each.bidRequest.Domain), 1)
            x.setdefault('AdSlotID'+"="+str(each.bidRequest.AdSlotID), 1)
            #x.setdefault('AdSlotWidth'+" "+str(each.bidRequest.AdSlotWidth), 1)
            #x.setdefault('AdSlotHeight'+" "+str(each.bidRequest.AdSlotHeight), 1)
            x.setdefault('AdSlotWidth', int(each.bidRequest.AdSlotWidth))
            x.setdefault('AdSlotHeight', int(each.bidRequest.AdSlotHeight))
            x.setdefault('AdSlotVisibility'+"="+str(each.bidRequest.AdSlotVisibility), 1)
            x.setdefault('AdSlotFormat'+"="+str(each.bidRequest.AdSlotFormat), 1)
            #x.setdefault('CreativeID', each['CreativeID'])
            #x.setdefault('Weekday', each['Weekday'])
            x.setdefault('hour'+"="+str(each.bidRequest.Timestamp[8:10]), 1)
            #x.setdefault('adid', each['adid'])
            #x.setdefault('UserTags', each['UserTags'])
            tags = str(each.bidRequest.UserTags).strip().split(",")
            for tag in tags:
                x.setdefault('UserTags'+"="+str(tag), 1)
            '''
            #x.setdefault('bid_price', float(each.bidResponse.BiddingPrice))
            count += 1
            #train_bid.append(float(each['bidding_price']))
            price = random.uniform(1., 301.)#300.)#float(each['bidding_price'])*random.uniform(0.01,0.999)
            train_bid.append(price)

            ctr.append(each['pCTR'])
            click.append(each['click'])
            train_data.append(x)
            #if float(each['bidding_price']) <= float(each['paying_price']):
            if price <= float(each['paying_price']):
                train_label.append(False)
            else:
                if allday:
                    epsmatrix.append(x)
                    epsbid.append(price)
                    if self.appro == '3' or self.appro == '1':
                        epswp.append(price - float(each['paying_price']))
                    else:
                        epswp.append(float(each['paying_price'])/price)
                train_label.append(True)

            train_win_p.append(float(each['paying_price']))
        if len(train_win_p) != 0:
            if f == None or f == 0:
                f = FeatureHasher(feature_size)
                train_outbid = f.fit_transform(train_data)
                if allday:
                    eps_outbid = f.transform(epsmatrix)
            else:
                train_outbid = f.transform(train_data)
            train_fdata = hstack([train_outbid, csr_matrix(np.reshape(train_bid, (-1, 1)))])

            if allday and (self.appro == '3' or self.appro == '1'):
                eps_foutbid = hstack([eps_outbid, csr_matrix(np.reshape(epsbid, (-1, 1)))])
            elif allday and (self.appro == '4' or self.appro == '2'):
                eps_foutbid = eps_outbid
            train_flabel = np.array(train_label)
            #print train_outbid.toarray()
            #print train_fdata.toarray()
            #print "*****Finish converting data: %s*****" %(timeit.default_timer()-default_time)
            if allday == False:
                if ctrtrain == True:
                    return train_fdata, train_outbid, train_flabel, train_win_p, train_bid, ctr, click, f
                else:
    
                    return train_fdata, train_outbid, train_flabel, train_win_p, train_bid, ctr, click, userid#0#f
            else:
                return train_fdata, train_outbid, train_flabel, train_win_p, train_bid, ctr, click, userid, eps_foutbid, epswp
        else:
            return 0, 0, 0, 0 ,0 ,0 ,0, 0
    def predict_price_hash(self, attr, feature_size, coef, w0, threshold):

        price_coef = coef[feature_size]
        pred_price = []
        cx = attr.tocoo()
        #print cx.row
        count = 0
        betaX = 0
        for r, i, xi in itertools.izip(cx.row, cx.col, cx.data):
            #print count, r
            if count != r:
                count = r
                betaX += w0
                price = ((-1)*(math.log(1/threshold-1)/math.log(math.exp(1)))-betaX)/price_coef
                if price < 0:
                    price = self.bid_floor
                if price > self.bid_upper:
                    price = self.bid_upper
                pred_price.append(price)
                betaX = 0
            betaX += coef[i]*xi
        betaX += w0
        price = ((-1)*(math.log(1/threshold-1)/math.log(math.exp(1)))-betaX)/price_coef
        if price < 0:
            price = self.bid_floor 
        if price >self.bid_upper:
            price = self.bid_upper
        pred_price.append(price)

        return pred_price
    #@profile(stream=fp)
    def given_recommend_price_hash(self, attr, feature_size, coef, w0, winp):
        
        price_coef = coef[feature_size]
        pred_price = []
        cx = attr.tocoo()
        count = 0
        betaX = 0
        prob = []
        label = []
        for r, i, xi in itertools.izip(cx.row, cx.col, cx.data):
            if count != r:
                count = r
                betaX += w0
                #eachprob = random.uniform(self.winrate_low, self.winrate_up)
                price = ((-1)*(math.log(1/random.uniform(self.winrate_low, self.winrate_up)-1)/math.log(math.exp(1)))-betaX)/price_coef
                #price = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX)/price_coef
                #prob.append(eachprob)
                if price < 0:
                    price = self.bid_floor
                if price > self.bid_upper:
                    price = self.bid_upper  
                pred_price.append(price)
                betaX = 0
            betaX += coef[i]*xi
        betaX += w0
        eachprob = random.uniform(self.winrate_low, self.winrate_up)
        #price = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX)/price_coef
        #prob.append(eachprob)
        price = ((-1)*(math.log(1/random.uniform(self.winrate_low, self.winrate_up)-1)/math.log(math.exp(1)))-betaX)/price_coef
        if price < 0:
            price = self.bid_floor
        if price > self.bid_upper:
            price = self.bid_upper
        pred_price.append(price)
        
        '''for each in xrange(0, len(pred_price)):
            if pred_price[each] > winp[each]:
                label.append(True)
            else:
                label.append(False)
        '''
        return pred_price#, self.ratio_eva_logloss(label, prob)
    def recommend_price_wr(self, attr, fs, coef, w0, wr):
        price_coef = coef[fs]
        cx = attr.tocoo()
        count = 0
        betaX = 0
        for r, i, xi in zip(cx.row, cx.col, cx.data):
            if count != r:
                print('get price')
                count = r
                betaX += w0
                price = ((-1)*(math.log(1/wr-1)/math.log(math.exp(1)))-betaX)/price_coef
            betaX += coef[i]*xi
        betaX += w0    
        price = ((-1)*(math.log(1/wr-1)/math.log(math.exp(1)))-betaX)/price_coef
        if price < 0:
            price = self.bid_floor
        if price > self.bid_upper:
            price = self.bid_upper
        return price
    def given_recommend_price_wr(self, attr, feature_size, coef, w0, winp, ad, pretrain_label, clicks, ctr = None, dens_ctr = None, th = .999, epsclf = None, ctrpro = 0., record = False, adid = None):
        thbudget =  ad.current_status
        #print "EPS: ", ad.eps
        price_coef = coef[feature_size]
        if self.appro == '3':
            eps_coef = epsclf.coef_[feature_size]
        else:
            eps_coef = None
        epsbid = []
        epsmatrix = csr_matrix((1, feature_size))
        epswp = []
        pred_price = []
        cx = attr.tocoo()
        eps = []
        count = 0
        betaX = 0
        eps_betax = 0
        prob = []
        cpm = []
        label = []
        etime = []
        eps_bug = []
        recordfirst = []
        a1 = []
        a2 = []
        a3 = []
        a4 = []
        a5 = []
        bs = []
        for r, i, xi in zip(cx.row, cx.col, cx.data):
            if count != r:
                st = timeit.default_timer()
                #count = r
                betaX += w0
                eps_betax += epsclf.intercept_[0]
                if pretrain_label:
                    eachprob = random.uniform(0.001, 0.999)#self.winrate_low, self.winrate_up)
                else:
                    if epsclf != None and self.appro == '4':
                        ad.eps = eps_betax#epsclf.predict(attr[count, :])[0]
                        #print eps_betax - ad.eps, eps_betax, ad.eps
                        #print ad.eps
                    #eachprob = ad.cal_winrate(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps, ad.stat)
                    if self.appro == '3' or self.appro == '4':
                        eachprob = ad.cal_winrate(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps, ad.stat, eps_coef, eps_betax)
                    else:
                        eachprob = ad.cal_winrate(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps, ad.stat)
                    eachprob = eachprob.real
                    #if ctr != None and norm.cdf(math.log(ctr[count]), loc = dens_ctr[0], scale = dens_ctr[1]) >= th:#random.uniform(0., 1.) <= ctr[count]:
                    #    eachprob = 0.999
                    eachprob += ctrpro*ctr[count]
                    eachprob = min(0.999, eachprob)
                price =float( ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX)/price_coef)

                ad.requests -= 1
                if ad.current_status > 10: #record joined bid
                    prob.append(eachprob)
                if price > ad.cpm/1000 and pretrain_label == False:
                    price = ad.cpm/1000
                if price <= self.bid_floor:
                    price = self.bid_floor

                if price > self.bid_upper:
                    price = self.bid_upper

                if (random.uniform(0, 1) > ad.pacing_rate or ad.current_status <= price) and pretrain_label == False: # not train and pacing rate
                    price = 0
                etime.append(timeit.default_timer()-st)
                if price != 0 or eachprob != 0.001:  #only record joined bid
                    pred_price.append(price)
                if pretrain_label == True: #price > winp[count] and pretrain_label == True:
                    ad.stat.append(winp[count])
                #if price > winp[count] and pretrain_label == True:
                #    epsmatrix = vstack([epsmatrix, attr[count, :]])
                #    epswp.append(price-winp[count])
                if price > winp[count] and pretrain_label == False:
                    cpm.append(winp[count])
                    epsbid.append(price)
                    epsmatrix = vstack([epsmatrix, attr[count, :]])
                    if self.appro == '3':
                        epswp.append(price-winp[count])
                    if self.appro == '4':
                        epswp.append(float(winp[count])/price)
                        a2.append((1.-ad.eps)*price)
                        a4.append((1.-epsclf.predict(attr[count, :]))*price)
                        a5.append(price-winp[count])
                    ad.budget_update(winp[count])
                    if self.appro == '1' or self.appro == '3':
                        eps.append(price-winp[count])
                    if self.appro == '2' or self.appro == '4':
                        eps.append(float(winp[count])/price)
                    ad.costs += winp[count]
                    ad.imps += 1
                    if clicks[count]:
                        ad.clicks += 1
                    #if record:
                    #    recordfirst.append(float(winp[count]))

                    #eps.append(1-float(price-winp[count])/price) #percentage
                    #print "win"
                #if random.uniform(self.winrate_low, self.winrate_up) <= 0.005 and pretrain_label == False: #sample and print results
                #if pretrain_label == False:
                #    print 'each ', ad.costs
                #    print 'each, ', ad.requests, ad.current_status#, winp[count], price, eachprob, betaX, price_coef
                count = r
                betaX = coef[i]*xi #0
                eps_betax = epsclf.coef_[i]*xi
                #if pretrain_label == False and record:
                #    bs.append(float(ad.costs))
            betaX += coef[i]*xi
            eps_betax += epsclf.coef_[i]*xi
        betaX += w0
        eps_betax += epsclf.intercept_[0]
        if pretrain_label:
            eachprob = random.uniform(0.001, 0.999)#(self.winrate_low, self.winrate_up)
        else:
            if epsclf != None and self.appro == '4':
                ad.eps = eps_betax#epsclf.predict(attr[count, :])[0]
            #    print 'eps, ', ad.eps
            #eachprob = ad.cal_winrate(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps, ad.stat, eps_coef, eps_betax)
            if self.appro == '3' or self.appro == '4':
                eachprob = ad.cal_winrate(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps, ad.stat, eps_coef, eps_betax)
            else:
                eachprob = ad.cal_winrate(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps, ad.stat)
            eachprob = eachprob.real
            #if ctr != None and norm.cdf(math.log(ctr[count]), loc = dens_ctr[0], scale = dens_ctr[1]) >= th:#random.uniform(0., 1.) <= ctr[count]:
            #    eachprob = 0.999
            eachprob += ctrpro*ctr[count]
            eachprob = min(0.999, eachprob)
        price = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX)/price_coef
        if ad.current_status > 10:
            prob.append(eachprob)
        #price = ((-1)*(math.log(1/random.uniform(self.winrate_low, self.winrate_up)-1)/math.log(math.exp(1)))-betaX)/price_coef

        if price > ad.cpm/1000 and pretrain_label == False:
            price = ad.cpm/1000

        if price <= self.bid_floor:
            price = self.bid_floor
        if price > self.bid_upper:
            price = self.bid_upper

        if (random.uniform(0, 1) > ad.pacing_rate or ad.current_status <= price) and pretrain_label == False:
            price = 0

        if price != 0 or eachprob != 0.001:
            pred_price.append(price)
        if pretrain_label == True: #price > winp[count] and pretrain_label == True:
            ad.stat.append(winp[count])
        #if price > winp[count] and pretrain_label == True:
        #    epsmatrix = vstack([epsmatrix, attr[count, :]])
        #    epswp.append(price-winp[count])
        if price > winp[count] and pretrain_label == False:
            cpm.append(winp[count])
            epsmatrix = vstack([epsmatrix, attr[count, :]])
            if self.appro == '3':
                epswp.append(price-winp[count])
            if self.appro == '2' or self.appro == '4':
                eps.append(float(winp[count])/price)
            ad.budget_update(winp[count])
            epsbid.append(price)
            if self.appro == '1' or self.appro == '3':
                eps.append(price-winp[count])
            if self.appro == '4':
                epswp.append(float(winp[count])/price)
                a2.append((1.-ad.eps)*price)
                a4.append((1.-epsclf.predict(attr[count, :]))*price)
                a5.append(price-winp[count])
            #eps.append(1-float(price-winp[count])/price) #percentage
            ad.costs += winp[count]
            ad.imps += 1
            if clicks[count]:
                ad.clicks += 1
            #if record:
            #    recordfirst.append(float(winp[count]))
        #if pretrain_label == False and record:
        #    bs.append(float(ad.costs))
        #    with open('pickle/ourbudgetspending_{0}.pickle'.format(adid), 'wb') as a:
        #        pickle.dump(bs, a)
        if pretrain_label == False: #calculate the eps by mean or percentage
            #ad.eps = sum(eps)/len(eps) #mean
            if len(eps) != 0: #geomean
                #ad.eps = pow(avg, 1./len(eps))
                ad.eps = scipy.stats.mstats.gmean(eps)
            else:
                print ("-*-*-*-*-*-*-*-*-*-*")
                ad.current_status = ad.budget_next(ad.current_status)
                return pred_price, prob, epsclf
        #if label == False:    
        #    print 'each ', ad.d, prob, epsclf = model.given_recommend_price_wr(hourly_outbid, model.feature_size, hourly_model.coef_[0], hourly_model.intercept_[0], hourly_winp, adv, label, click, ctrlist, dens_ctr, th, epsclf, ct, record = record)costs
        #    print 'each, ', ad.requests, ad.current_status#, winp[count], price, eachprob, betaX, price_coef
        #if epsclf == None:
        #    epsclf = SGDRegressor(penalty='l2', alpha = 0.01, fit_intercept=True, n_iter = 1, shuffle = False, warm_start = True, learning_rate = 'constant', eta0 = 0.0000005)#Ridge()
        #epsclf.partial_fit(epsmatrix[1:, :], epswp)
        if pretrain_label == False:
            if self.appro == '3' or self.appro == '4':
                if epsclf == None:
                    epsclf = SGDRegressor(penalty='l2', alpha = 0.01, fit_intercept=True, n_iter_no_change  = 1, shuffle = False, warm_start = True, learning_rate = 'constant', eta0 = 0.000000005)#Ridge()
                if self.appro == '3':
                    eps_foutbid = hstack([epsmatrix[1:, :], csr_matrix(np.reshape(epsbid, (-1, 1)))])
                else:
                    eps_foutbid = epsmatrix[1:, :]
                if self.appro == '3':
                    eps_bug = epsclf.predict(eps_foutbid)
                    print ('eps comp, ', ad.eps, np.mean(eps_bug), scipy.stats.mstats.gmean(eps_bug), 'MSE, ', metrics.mean_squared_error(epswp, [ad.eps]*len(epswp)), metrics.mean_squared_error(epswp, eps_bug))
                if self.appro == '4':
                    print ('eps comp, ', ad.eps, np.mean(eps_bug), scipy.stats.mstats.gmean(eps_bug), 'MSE, ', metrics.mean_squared_error(a5, a2), metrics.mean_squared_error(a5, a4))
                epsclf.partial_fit(eps_foutbid, epswp)
            ad.current_status = ad.budget_next(ad.current_status)
            print ('average execution time, ', np.mean(etime))
            print ("current rest: ",ad.current_status, 'average winrate', scipy.stats.mstats.gmean(prob), "total cost: ", sum(cpm), "allocated budget: ", thbudget, "cpm: ", sum(cpm)/len(cpm), "imp: ", len(cpm), 'stat: ' , ad.stat)
            #if record:
            #    with open('pickle/ours_{0}_{1}_{2}.pickle'.format(str(ad.budget), str(self.cpm/1000.), adid), 'wb') as d:
            #        pickle.dump(recordfirst, d)
        return pred_price, prob, epsclf

    '''def given_recommend_price_wr_performance(self, attr, feature_size, coef, w0, winp, ad, pretrain_label, ctr, click):
        print "New budget: ", ad.current_status
        print "EPS: ", ad.eps
        price_coef = coef[feature_size]
        pred_price = []
        cx = attr.tocoo()
        eps = []
        count = 0
        betaX = 0
        prob = []
        cpm = []
        label = []
        real_click = []
        unit_click = ad.unit_click
        for r, i, xi in itertools.izip(cx.row, cx.col, cx.data):
            if count != r:
                #count = r
                betaX += w0

                if pretrain_label:
                    eachprob = random.uniform(self.winrate_low, self.winrate_up)
                else:
                    #eachprob = ad.cal_winrate_performance(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps, ctr[count]) #maxcpc
                    eachprob = ad.cal_winrate(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps) # with budget constrain
                price = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX)/price_coef
                
                #if pretrain_label == False: #with budget constrain
                #    if unit_click > 0:
                #        if ad.current_status*ctr[count]/unit_click > price:
                #            price = random.uniform(price, ad.current_status*ctr[count]/unit_click)
                #    else:
                #price = ad.current_status*ctr[count]/ad.unit_click
                price = ctr[count]*ad.cpc #max ecpc
                ad.requests -= 1

                if ad.current_status > 10: #record joined bid
                    prob.append(eachprob)

                if price <= 0:
                    price = self.bid_floor
                if price > self.bid_upper:
                    price = self.bid_upper

                if (random.uniform(0, 1) > ad.pacing_rate or ad.current_status <= 0) and pretrain_label == False: # not train and pacing rate
                    price = 0

                if price != 0 or eachprob != 0.001:  #only record joined bid
                    pred_price.append(price)

                if price > winp[count] and pretrain_label == False:
                    cpm.append(winp[count])
                    real_click.append(click[count])
                    ad.budget_update(winp[count])
                    eps.append(price-winp[count])
                    ad.costs += winp[count]
                    ad.imps += 1
                    if click[count]:
                        ad.clicks += 1
                        if unit_click > 0:
                            unit_click -= 1
                
                    #eps.append(1-float(price-winp[count])/price) #percentage
                    #print "win"
                if random.uniform(self.winrate_low, self.winrate_up) <= 0.005 and pretrain_label == False: #sample and print results
                    print count, ad.current_status, winp[count], price, eachprob, ctr[count], click[count] 
                count = r
                betaX = coef[i]*xi #0
            betaX += coef[i]*xi

        betaX += w0

        if pretrain_label:
            eachprob = random.uniform(self.winrate_low, self.winrate_up)
        else:
            #eachprob = ad.cal_winrate_performance(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps, ctr[r])

            eachprob = ad.cal_winrate(self.winrate_low, self.winrate_up, ad.current_status, coef, self.feature_size, ad.pacing_rate, ad.requests, count, betaX, ad.eps) # with budget constrain
        price = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX)/price_coef

        #if pretrain_label == False: #with budget constrain
        #    if unit_click > 0:
        #        if ad.current_status*ctr[count]/unit_click > price:
        #            price = random.uniform(price, ad.current_status*ctr[count]/unit_click)
        #    else:
        #price = ad.current_status*ctr[count]/ad.unit_click
        price = ctr[count]*ad.cpc #max ecpc



        if ad.current_status > 10:
            prob.append(eachprob)
        #price = ((-1)*(math.log(1/random.uniform(self.winrate_low, self.winrate_up)-1)/math.log(math.exp(1)))-betaX)/price_coef

        if price <= 0:
            price = self.bid_floor
        if price > self.bid_upper:
            price = self.bid_upper

        if (random.uniform(0, 1) > ad.pacing_rate or ad.current_status <= 0) and pretrain_label == False:
            price = 0

        if price != 0 or eachprob != 0.001:
            pred_price.append(price)

        if price > winp[count] and pretrain_label == False:
            cpm.append(winp[count])
            real_click.append(click[count])
            ad.budget_update(winp[count])
            eps.append(price-winp[count])
            ad.costs += winp[count]
            ad.imps += 1
            if click[count]:
                ad.clicks += 1

            #eps.append(1-float(price-winp[count])/price) #percentage

        if pretrain_label == False: #calculate the eps by mean or percentage
            #ad.eps = sum(eps)/len(eps) #mean
            if len(eps) != 0: #geomean
                #ad.eps = pow(avg, 1./len(eps))
                ad.eps = scipy.stats.mstats.gmean(eps)
            else:
                print "-*-*-*-*-*-*-*-*-*-*"
        #if sum(real_click) == 0: # to prevent dividing zero
        #    real_click.append(-1)
        print count, ad.current_status, winp[count], price, eachprob, ctr[count], click[count]
        if pretrain_label == False:
            ad.current_status = ad.budget_next(ad.current_status)
            print "current rest: ",ad.current_status, 'average winrate', scipy.stats.mstats.gmean(prob), "total cost: ", sum(cpm), "imps: ", len(cpm), 'clicks: ', sum(real_click), "avaiable clicks: ", sum(click)#, 'ecpc: ', float(sum(cpm))/sum(real_click)
        return pred_price, prob


    def blending_given_recommend_price_hash(self, attr, feature_size, coef0, w00, winp, coef1, w10, alpha):

        price_coef0 = coef0[feature_size]
        price_coef1 = coef1[feature_size]

        pred_price = []
        cx = attr.tocoo()
        count = 0
        betaX0 = 0
        betaX1 = 0
        prob = []
        label = []
        for r, i, xi in itertools.izip(cx.row, cx.col, cx.data):
            if count != r:
                count = r

                betaX0 += w00
                betaX1 += w10

                eachprob = random.uniform(self.winrate_low, self.winrate_up)
                price0 = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX0)/price_coef0
                price1 = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX1)/price_coef1
                prob.append(eachprob)
                price = alpha*price0+(1.-alpha)*price1
                if price < 0:
                    price = self.bid_floor
                if price > self.bid_upper:
                    price = self.bid_upper
                pred_price.append(price)
                betaX0 = 0
                betaX1 = 0
            betaX0 += coef0[i]*xi
            betaX1 += coef1[i]*xi
        betaX0 += w00
        betaX1 += w10
        eachprob = random.uniform(self.winrate_low, self.winrate_up)
        price0 = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX0)/price_coef0
        price1 = ((-1)*(math.log(1/eachprob-1)/math.log(math.exp(1)))-betaX1)/price_coef1

        prob.append(eachprob)
        price = alpha*price0+(1.-alpha)*price1
        if price < 0:
            price = self.bid_floor
        if price > self.bid_upper:
            price = self.bid_upper
        pred_price.append(price)

        for each in xrange(0, len(pred_price)):
            if pred_price[each] > winp[each]:
                label.append(True)
            else:
                label.append(False)

        return pred_price#, self.ratio_eva_logloss(label, prob)
    '''
    def cpm_eva(self, pred_price, win_p):

        total = []

        for i in range(0, len(pred_price)):
            if pred_price[i] > win_p[i]:
                total.append(win_p[i])
        if len(total) != 0:
            print ("CPM: ", sum(total)/len(total), len(total), "Origin CPM: ", sum(win_p)/len(win_p), len(win_p))
        else:
            print ("CPM: ", sum(total)/(len(total)+1), len(total), "Origin CPM: ", sum(win_p)/len(win_p), len(win_p))
    def ratio_eva_hash(self, data_outbid, feature_size, coef, intercept, win_price):

        i = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
        c = []
        avg_logloss = []
        for r in i:
            p_price = self.predict_price_hash(data_outbid, feature_size, coef, intercept, r)
            t = 0
            f = 0
            tl = 0
            fl = 0
            #label = []
            for each in range(0, len(win_price)):
                if p_price[each] > win_price[each]:
                    t += 1.
                    #label.append(True)
                else:
                    f += 1.
                    #label.append(False)
            #temp_log = self.ratio_eva_logloss(label, r)
            #print "Log loss", temp_log
            #avg_logloss.append(temp_log)
            c.append(t/len(win_price))
    
        print ("Ratio RMSE: ", pow(metrics.mean_squared_error(i, c),0.5))
        print ("Slope:", linregress(i, c)[0])
        return pow(metrics.mean_squared_error(i, c),0.5), linregress(i, c)[0]#, np.mean(avg_logloss)
    def ratio_eva_logloss(self, true_label, prob):
        summary = 0
        for i in range(0, len(true_label)):
            if true_label[i] == True:
                summary += (-1)*(math.log(prob[i])/math.log(math.exp(1)))
            else:
                summary += (-1)*(math.log(1-prob[i])/math.log(math.exp(1)))
        logloss = summary/len(true_label)
        #prob = [ratio]*len(true_label)
        print ("Log loss: ", logloss)#,metrics.log_loss(true_label, prob) 
        return logloss#metrics.log_loss(true_label, prob)
            
    def train_model_dictovec(self, train, label, coef = None):
    
        logreg = LogisticRegression(penalty='l2', dual=False)
        logreg.fit(train, label)
        
        return logreg

    def train_model_hash(self, train, label, model = None, init = True, SGD = False):
        
        if model == None:
            if SGD:
                clf = SGDClassifier(loss='log', penalty='l2', alpha = 0.01, fit_intercept=True, n_iter_no_change  = 1, shuffle = False, warm_start = True, learning_rate = 'constant', eta0 = 0.000000005)
                print ("SGD")
            else:
                clf = LogisticRegression(penalty='l2', dual=False)
            clf.fit(train, label)
            print ("all")
            return clf
        else:
        
            if init:
        
                clf = SGDClassifier(loss='log', penalty='l2', alpha = 0.01, fit_intercept=True, n_iter_no_change  = 1, shuffle = False, warm_start = True, learning_rate = 'constant', eta0 = 0.000000005)
                clf.fit(train, label, model.coef_, model.intercept_)# [True, False])
                print ("part")
                return clf
            else:
                model.partial_fit(train, label, [True, False])
                return model
    def train_smartmodel_hash(self, train, label, alpha1 = 0.001):

        reg = Ridge(alpha = alpha1)
        reg.fit(train, label)
        return reg

        '''if model == None:
            reg = SGDRegressor(alpha = 1000, learning_rate = 'constant', eta0 = 0.000000005)
            reg.fit_transform(train, label)
            return reg
        else:
            model.partial_fit(train, label) 
            return model
        '''
    def combine_csrvstack(self, matrix1, matrix2):
        new_data = np.concatenate((matrix1.data, matrix2.data))
        new_indices = np.concatenate((matrix1.indices, matrix2.indices))
        new_ind_ptr = matrix2.indptr + len(matrix1.data)
        new_ind_ptr = new_ind_ptr[1:]
        new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))
        return csr_matrix((new_data, new_indices, new_ind_ptr))
