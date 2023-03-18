from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
from pymongo import MongoClient
import datetime, csv
import scipy.stats as ss
import onoff
from sklearn import metrics
dates = [("10","01"), ("10","02"), ("10","03"), ("10","04"), ("10","05"), ("10","06"), ("10","07")]
#dates = [("06","06"), ("06","07"), ("06","08"), ("06","09"), ("06","10"), ("06","11"), ("06","12")]#, ("10","19"), ("10","20"), ("10","21"), ("10","22"), ("10","23"), ("10","24"), ("10","25"), ("10","26"), ("10","27")]
#dates = ["19", "20", "21", "22", "23", "24", "25", "26", "27"]
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
def MAPE(true, pred):

    result = 0
    for each in xrange(0, len(pred)):
        result += float(abs(pred[each]-true[each]))/true[each]
    return result/len(pred)
class Traffic:
    
    def __init__(self, traindays, aid):
        self.dv = DictVectorizer(sparse=False)
        self.lr = Ridge(alpha = pow(10,2))
        #self.lr = LinearRegression()
        self.traindays = traindays
        self.hour_window = [[0 for col in xrange(24)] for row in range(traindays+2)]
        temp = []
        for hour in xrange(0, 24):
            temp.append({'hour':str(hour), 'lastHour':0, 'lastDay':0})
        self.dv.fit(X=temp)
        del temp
        self.aid = aid
        self.first_train()

    def preprocess(self):
        #print self.hour_window
        #build training set for the past trainingdays
        trainX = []
        trainY = []
        for day in xrange(len(self.hour_window)- self.traindays-1, len(self.hour_window)-1):                 
            for hour in xrange(0, 24):
                temp = {}                                      
                temp['hour'] = str(hour)
                temp['lastDay'] = self.geomean(len(self.hour_window), hour)#self.hour_window[day-1][hour]
                if hour > 0:
                    temp['lastHour'] = self.check_abnormal(3, len(self.hour_window), hour-1)#self.hour_window[day][hour-1]
                else:
                    temp['lastHour'] = self.check_abnormal(3, len(self.hour_window), 23)#self.hour_window[day-1][23]
                trainX.append(temp)
                trainY.append(self.hour_window[day][hour])
        #pop the past fourth day from the window
        #self.hour_window.pop(0)                 
        #new_row = [0]*24
        #self.hour_window.append(new_row)                                         
        return self.dv.transform(trainX), trainY              
    
    def check_abnormal(self, tol, history, prehour):
        
        temp = []
        if prehour != 23:
            for i in xrange(history-2, history-1-self.traindays-1, -1):
                if self.hour_window[i][prehour] != 0:
                    temp.append(self.hour_window[i][prehour])
        elif prehour == 23:
            for i in xrange(history-3, history-1-self.traindays-2, -1):
                if self.hour_window[i][prehour] != 0:
                    temp.append(self.hour_window[i][prehour])
                
        avg = np.mean(temp)
        std = np.std(temp)
        if float(abs(self.hour_window[len(self.hour_window)-1][prehour]-avg))/std > tol:
            if self.hour_window[len(self.hour_window)-1][prehour] != 0:
                temp.append(self.hour_window[len(self.hour_window)-1][prehour])
            return ss.gmean(temp)
        else:
            return self.hour_window[len(self.hour_window)-1][prehour]
    def geomean(self, history, hour):
        
        temp = []
        for i in xrange(history-2, history-1-self.traindays-1, -1):
            if self.hour_window[i][hour] != 0:
                temp.append(self.hour_window[i][hour])
        return ss.gmean(temp)
    def first_preprocess(self):
        #print self.hour_window
        #build training set for the past trainingdays
        trainX = []
        trainY = []
        for day in xrange(len(self.hour_window)- self.traindays, len(self.hour_window)-1):
            for hour in xrange(0, 24):
                temp = {}
                temp['hour'] = str(hour)
                temp['lastDay'] = self.hour_window[day-1][hour]
                if hour > 0:
                    temp['lastHour'] = self.hour_window[day][hour-1]
                else:
                    temp['lastHour'] = self.hour_window[day-1][23]
                trainX.append(temp)
                trainY.append(self.hour_window[day][hour])

        #pop the past fourth day from the window
        #self.hour_window.pop(0)
        #new_row = [0]*24
        #self.hour_window.append(new_row)

        return self.dv.transform(trainX), trainY
    def first_train(self):
        
        client = MongoClient('140.113.216.125', 11225)
        db = client.DoubleClick
        
        for i in xrange(0, 3):
            for h in xrange(0, len(hours)):
                month = dates[i][0]
                date = dates[i][1]
                hour = hours[h]
                time1 = datetime.datetime.strptime('2016-'+month+'-'+date+' '+hour, '%Y-%m-%d %H')
                time2 = time1 + datetime.timedelta(hours=1)
                if self.aid == None:
                    cursor = db.doubleclick.find({'dateTime':{"$gte": time1, '$lt': time2}, "bidResponse.price": {"$gt": 0.}, "winEvent.price": {"$gt":0.}})
                else:
                    cursor = db.doubleclick.find({'dateTime':{"$gte": time1, '$lt': time2}, "bidResponse.price": {"$gt": 0.}, "winEvent.price": {"$gt":0.}, 'bidResponse.advertiserId':self.aid})
                self.hour_window[1+i][h] = cursor.count()
        client.close()
        trainX, trainY = self.first_preprocess()
        self.lr.fit(trainX, trainY)

    def daily_train(self):

        trainX, trainY = self.preprocess()
        self.lr = Ridge()
        self.lr.fit(trainX, trainY)
        self.hour_window.pop(0)
        new_row = [0]*24
        self.hour_window.append(new_row)

    def hourly_update(self, req, hour):
        self.hour_window[len(self.hour_window)-1][hour] = req
    
    def predict(self, hour):
        temp = {}
        temp['hour'] = str(hour)
        temp['lastDay'] = self.geomean(len(self.hour_window), hour)#self.hour_window[day-1][hour]
        if hour > 0:
            temp['lastHour'] = self.check_abnormal(1, len(self.hour_window), hour-1)#self.hour_window[day][hour-1]
        else:
            temp['lastHour'] = self.check_abnormal(1, len(self.hour_window), 23)#self.hour_window[day-1][23]
        predictX = self.dv.transform(temp)
        return self.lr.predict(predictX)[0]
'''
if __name__ == '__main__':
    pretrain_days = 4 
    traffic = Traffic(pretrain_days)
    model = onoff.Model(0.999)
    new_day = -1
    req = 0
    pre = []
    gro = []
    for i in xrange(0, len(dates)):
        if i < pretrain_days:
            label = True
        else:
            label = False
        if i != 0:
            for h in xrange(0, len(hours)):
                
                if new_day != i and i != pretrain_days and label == False:
                    new_day = i
                    traffic.daily_train()
                if label == False:
                    req = int(traffic.predict(int(hours[h])))

                    request_ground = model.get_data(dates[i][0], dates[i][1], hours[h], 'hourly')
                    traffic.hourly_update(request_ground.count(), int(hours[h]))
                    print dates[i],hours[h], 'predict', req, 'truth', request_ground.count()
                    pre.append(req)
                    gro.append(request_ground.count())

    print 'RMSE', pow(metrics.mean_squared_error(gro, pre),0.5), 'MAPE', MAPE(gro, pre), sum(pre), sum(gro), float(sum(pre)-sum(gro))/sum(gro)
'''
