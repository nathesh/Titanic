
def train(train_data):
	print 'Training'
	NaiveBayes = GaussianNB()
	NaiveBayes = NaiveBayes.fit( train_data[0::,1::], train_data[0::,0])
	return NaiveBayes
print 'Predicting...'
output = NaiveBayes.predict(test_data).astype(int)
predictions_file = open("myfirstNB.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

def run(train_data,):
	print 'Naive Bayes'
	NaiveBayes = train(train_data)