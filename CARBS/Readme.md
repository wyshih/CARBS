# CARBS
package dependency: numpy, scipy, sklearn, pymongo, kerasrl, gym, keras
steps:
1. construct a mongo db and create a table to store iPinyou dataset which can be found on internet.
2. execute the timeslot_withpredicnoupdate.py file in wrmodel/ipin to build the ctr prediction model and store the clusters.
3. execute switchtestenv.py to start training CARBS
