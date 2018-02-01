import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.metrics import accuracy_score
from neatdata.neatdata import *
from sklearn.metrics import confusion_matrix
import pickle

class ModelPipeline:

    def __init__(self):
        self.indexColumns, self.skipColumns = None, None
        self.neatData =  NeatData()
        self.className = 'class' # Edit: Replace class with the Y column name
        self.indexColumns = [] # Edit: Optionally add column names
        self.skipColumns = [] # Edit: Optionally add column names


    def execute(self):
        trainX, testX, trainY, testY = self._getDatasetFrom________() # Edit: choose one of two functions
        cleanTrainX, cleanTrainY, cleanTestX, cleanTestY = self._cleanDatasets()
    

    def _getDatasetFromOneFile(self):
        df = pd.read_csv('iris.csv') # Edit: Your dataset
        trainX, testX, trainY, testY = train_test_split(df.drop([self.className], axis=1),
                                                         df[self.className], train_size=0.75, test_size=0.25)
        return trainX, testX, trainY, testY

    def _getDatasetFromTwoFiles(self):
        trainingDf = pd.read_csv('train_iris.csv') # Edit: Your training dataset
        testDf = pd.read_csv('test_iris.csv') # Edit: Your test dataset
        trainX = trainingDf.drop([self.className], axis=1)
        trainY = trainingDf[self.className]
        testX = testDf.drop([self.className], axis=1)
        testY = testDf[self.className]
        return trainX, testX, trainY, testY

    def _cleanDatasets(self):
        cleanTrainX, cleanTrainY = self.neatData.cleanTrainingDataset(trainX, trainY, indexColumns, skipColumns)
        cleanTestX = self.neatData.cleanTestDataset(testX)
        cleanTestY = self.neatData.convertYToNumbersForModeling(testY)
        return cleanTrainX, cleanTrainY, cleanTestX, cleanTestY

    def _modelFit(self):
		exported_pipeline = make_pipeline(
		    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.9000000000000001, min_samples_leaf=9, min_samples_split=5, n_estimators=100, subsample=0.4)),
		    StackingEstimator(estimator=GaussianNB()),
		    GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.6000000000000001, min_samples_leaf=5, min_samples_split=7, n_estimators=100, subsample=1.0)
		)
		
		exported_pipeline.fit(cleanTrainX, cleanTrainY)
		results = exported_pipeline.predict(cleanTestX)

    def printModelScores(self):
        print("Confusion Matrix:")
        print(confusion_matrix(cleanTestY, results))
        print(accuracy_score(cleanTestY, results))

    def createTrainedModelPipelineFile(self):
        def save_object(obj, filename):
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
        save_object(self, 'ModelPipeline.pkl')
        with open('trainedmodelpipeline.py', 'w') as fileOut:
            fileOut.write("""

import pandas as pd
import pickle

class TrainedModelPipeline:

    def __init__(self):
        self.modelPipeline = None
        self.cleanTestX = None

    def execute(self):
        with open('ModelPipeline.pkl', 'rb') as input:
            self.modelPipeline = pickle.load(input)
        testX = self._getDataset()
        self.cleanTestX = self._cleanDataset(testX)
        results = self._predict()
        resultsDf = self._concatenatePredictionsToDataframe(results)
        self._saveResultsAsCSV(resultsDf)
        print("Done.  Created results.csv")

    def _getDataset(self):
        return pd.read_csv('test_iris.csv') # Edit: Your dataset

    def _cleanDataset(self, testX):
        return neatData.cleanTestDataset(testX)

    def _predict(self):
        results = exported_pipeline.predict(self.cleanTestX)
        return neatData.convertYToStringsOrNumbersForPresentation(results)

    def _concatenatePredictionsToDataframe(self, results):
        resultsDf = pd.DataFrame(results)
        return pd.concat([testX, resultsDf], axis=1)

    def _saveResultsAsCSV(self, resultsDf):
        resultsDf.to_csv('./results.csv')

trainedModelPipeline = new TrainedModelPipeline()
trainedModelPipeline.execute()        
""")

modelPipeline = new ModelPipeline()
modelPipeline.execute()


