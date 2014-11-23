from sklearn import svm
import numpy as np
import csv as csv

def train(train_data):
	print 'Training'
	SVM =  svm.SVC()
	SVM = SVM.fit(train_data[0::,1::], train_data[0::,0])
	return SVM

def predict(SVM,test_data):
	print 'Predicting...'
	output = SVM.predict(test_data).astype(int)
	return output
def output(ids,output):
	predictions_file = open("output/myfirstSVM.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, output))
	predictions_file.close()
	print 'Done.'

def run(train_data,test_data,ids):
	print 'SVM'
	ret_train = train(train_data)
	predict_output = predict(ret_train,test_data)
	output(ids,predict_output)