from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
from pymongo import MongoClient
import datetime, csv, onoff
from sklearn import metrics 
class Traffic:
	def __init__(self):
		#self.received = 0 #memorizing traffic in a slot
		self.dv = DictVectorizer(sparse=False) 
		self.lr = Ridge() #prediction model
		self.hour_window = [[0 for col in xrange(24)] for row in range(4)]
		
		#make dictvetorizer
		temp = []
		for hour in xrange(0, 24):
			temp.append({'hour':str(hour), 'lastHour':0, 'lastDay':0})
		self.dv.fit(X=temp)
		del temp
		
		#training 
		self.first_train()
	
	def preprocess(self, trainingdays=3):
		print self.hour_window
		#build training set for the past trainingdays
		trainX = []
		trainY = []
		for day in xrange(4-trainingdays, 4):
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
		self.hour_window.pop(0)
		new_row = [0]*24
		self.hour_window.append(new_row)
		
		return self.dv.transform(trainX), trainY
	
	def smoothing(self, received, hour):
		print received
		if received > 0:
			return np.power(received*self.hour_window[0][hour]*self.hour_window[1][hour]*self.hour_window[2][hour], 0.25)							
		else:
			return np.power(self.hour_window[0][hour]*self.hour_window[1][hour]*self.hour_window[2][hour], 1/3.0)
		
	def hourly_update(self, received, hour):
		#smooth outliers if necessary
		self.hour_window[3][hour] = self.smoothing(received, hour)
		
	
	def daily_train(self, alpha=1):
		#re-train the linear regression model when the day starts
		trainX, trainY = self.preprocess(trainingdays=3)
		self.lr.fit(trainX, trainY)
	
	def predict(self, hour):
		#predict when the hour starts
		if hour > 0:
			day = 3
		else:
			day = 2
		temp = {'hour':str(hour), 'lastHour':self.hour_window[day][(hour-1)%24], 'lastDay':self.hour_window[2][hour]}
		print temp
		predictX = self.dv.transform(temp)
				
		return self.lr.predict(predictX)[0]
	
	def first_train(self):
		#setting mongo, for season2
		client = MongoClient('140.113.216.125', 11225)
		db = client.iPinYou_allinone
		time1 = datetime.datetime.strptime('2013-6-6','%Y-%m-%d')
		time2 = datetime.datetime.strptime('2013-6-9','%Y-%m-%d')
		cursor = db.ipinyou.find({'timestamp':{"$gte": time1, '$lt': time2}, "bidding_price": {"$gt": 0.}, "paying_price": {"$gt":0.}})
		
		
		
		#collect initial training set
		for each in cursor:
			day = each['timestamp'].day
			hour = each['timestamp'].hour
			self.hour_window[day-5][hour] += 1
		
		client.close()
		
		trainX, trainY = self.preprocess(trainingdays=2)
		
		self.lr.fit(trainX, trainY)
dates = [("06","06"), ("06","07"), ("06","08"), ("06","09"), ("06","10"), ("06","11"), ("06","12")]#, ("10","19"), ("10","20"), ("10","21"), ("10","22"), ("10","23"), ("10","24"), ("10","25"), ("10","26"), ("10","27")]          
#dates = ["19", "20", "21", "22", "23", "24", "25", "26", "27"]                                                   
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"] 
def MAPE(true, pred):                                                                                             
                                                                                                                  
    result = 0                                                                                                    
    for each in xrange(0, len(pred)):                                                                             
        result += float(abs(pred[each]-true[each]))/true[each]                                                    
    return result/len(pred) 
if __name__ == '__main__':                                                                                        
    pretrain_days = 3                 
    traffic = Traffic()                                                                              
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
		
