from sklearn.ensemble import RandomForestClassifier

def train(train_data):
	print 'Training...'
	forest = RandomForestClassifier(n_estimators=100)
	forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
	return forest
def predict(test_data):
	print 'Predicting...'
	return output_text = forest.predict(test_data).astype(int)

def output(ids,output_text):
	predictions_file = open("myfirstforest.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, output_text))
	predictions_file.close()
	print 'Done.'

def run(train_data,test_data,ids):
	print 'Start RandomForests...'
	forest = train(train_data)
	output_text = predict(test_data)
	output(ids,output)