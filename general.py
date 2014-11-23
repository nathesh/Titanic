import pandas as pd
import numpy as np
import csv as csv
import NaiveBayes
import randomforest
import SVM
def clean_train_data(train_df):
    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.

    # female = 0, Male = 1
    train_df['Gender'] = train_df['Sex'].map(
        {'female': 0, 'male': 1}).astype(int)
    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2"
    # is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:
        train_df.Embarked[
            train_df.Embarked.isnull()] = train_df.Embarked.dropna().mode().values
    # All the ages with no data -> make the median of all Ages
    median_age = train_df['Age'].dropna().median()
    if len(train_df.Age[train_df.Age.isnull()]) > 0:
        train_df.loc[(train_df.Age.isnull()), 'Age'] = median_age

    # determine all values of Embarked,
    Ports = list(enumerate(np.unique(train_df['Embarked'])))
    # set up a dictionary in the form  Ports : index
    Ports_dict = {name: i for i, name in Ports}
    train_df.Embarked = train_df.Embarked.map(lambda x: Ports_dict[x]).astype(
        int)     # Convert all Embark strings to int

    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and
    # filled it to Gender)
    train_df = train_df.drop(
        ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    # The data is now ready to go. So lets fit to the train, then predict to the test!
    # Convert back to a numpy array
    train_data = train_df.values
    return train_data,Ports_dict


def clean_test_data(Ports_dict,test_df):
    # I need to do the same with the test data now, so that the columns are the same as the training data
    # I need to convert all strings to integer classifiers:
    # female = 0, Male = 1
    test_df['Gender'] = test_df['Sex'].map(
        {'female': 0, 'male': 1}).astype(int)

    # Embarked from 'C', 'Q', 'S'
    # All missing Embarked -> just make them embark from most common place
    if len(test_df.Embarked[test_df.Embarked.isnull()]) > 0:
        test_df.Embarked[
            test_df.Embarked.isnull()] = test_df.Embarked.dropna().mode().values
    # Again convert all Embarked strings to int
    test_df.Embarked = test_df.Embarked.map(
        lambda x: Ports_dict[x]).astype(int)

    # All the ages with no data -> make the median of all Ages
    median_age = test_df['Age'].dropna().median()
    if len(test_df.Age[test_df.Age.isnull()]) > 0:
        test_df.loc[(test_df.Age.isnull()), 'Age'] = median_age

    # All the missing Fares -> assume median of their respective class
    if len(test_df.Fare[test_df.Fare.isnull()]) > 0:
        median_fare = np.zeros(3)
        # loop 0 to 2
        for f in range(0, 3):
            median_fare[f] = test_df[
                test_df.Pclass == f + 1]['Fare'].dropna().median()
        # loop 0 to 2
        for f in range(0, 3):
            test_df.loc[(test_df.Fare.isnull()) & (
                test_df.Pclass == f + 1), 'Fare'] = median_fare[f]

    # Collect the test data's PassengerIds before dropping it
    ids = test_df['PassengerId'].values
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and
    # filled it to Gender)
    test_df = test_df.drop(
        ['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    test_data = test_df.values
    return ids, test_data

def clean_data(train_df,test_df):
	train_data,Ports_dict = clean_train_data(train_df)
	ids, test_data = clean_test_data(Ports_dict,test_df)
	return train_data,test_data,ids

if __name__ == '__main__':
    # Data cleanup
    # Load the test file into a dataframe
	test_df = pd.read_csv('data/test.csv', header=0)
    # Load the train file into a dataframe
	train_df = pd.read_csv('data/train.csv', header=0)
	train_data,test_data,ids = clean_data(train_df,test_df)
	SVM.run(train_data,test_data,ids)
    
