# This script will be able to take in a pickle dataset and output the dataset with outliers removed


# imports
import pandas as pd
import numpy as np
import pickle


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest


# set a random state? 


# grab pickle file
data = pickle.load(open('redis_vector_dump.pkl', 'rb'))
print(f"The pickle file is type {type(data)}.")


# explore pickle file and make it into pandas df to easily feed into the IsolationForest
df = pd.DataFrame.from_dict(data, orient='index')


# set up outlier detection algorithm (in pipeline)
model = IsolationForest(n_estimators = 10)



# can try making a pipeline later



# run pickle dataset through it
model.fit(df)


# try to see if I can do a "predict"
predictions = model.predict(df)

# turn predictions into a DF to see the output
pred_df = pd.DataFrame(predictions)

# print statements on how many outliers were detected
print(f"The model predicted the following outliers:")
print(pred_df.value_counts(normalize=True))


# add the prediction output as a new column into the dataset



# filter out the -1 (outliers) from the dataset


# create new dataset without outliers, possibly if statement



# output new dataset to new pickle file