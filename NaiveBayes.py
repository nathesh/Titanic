from sklearn.naive_bayes import GaussianNB
import numpy as np
import csv as csv


def train(train_data):
	print 'Training'
	NaiveBayes = GaussianNB()
	NaiveBayes = NaiveBayes.fit( train_data[0::,1::], train_data[0::,0])
	return NaiveBayes

def predict(NaiveBayes,test_data):
	print 'Predicting...'
	output = NaiveBayes.predict(test_data).astype(int)
	return output

def output(output,ids):
	predictions_file = open("output/myfirstNB.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, output))
	predictions_file.close()
	print 'Done.'

def run(train_data,test_data,ids):
	print 'Naive Bayes'
	NaiveBayes = train(train_data)
	predict_output = predict(NaiveBayes,test_data)
	output(predict_output,ids)
