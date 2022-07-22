# This script will be able to take in a pickle dataset and output the dataset with outliers removed


# imports
import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest


# grab pickle file
data = pickle.load(open('redis_vector_dump.pkl', 'rb'))
print(f"The pickle file is type {type(data)} and has {len(data)} records.")


# explore pickle file and make it into pandas df to easily feed into the IsolationForest
df = pd.DataFrame.from_dict(data, orient='index')


# set up outlier detection algorithm (in pipeline)
model = IsolationForest(n_estimators = 10, random_state=1)


# fit the model on the data in pandas df format
model.fit(df)


# output predictions from the model on whether each row is an outlier or not
predictions = model.predict(df)

# turn predictions into a pandas DF to see the output
pred_df = pd.DataFrame(predictions)

# print statements on % outliers detected
print(f"The model predicted the following percent of inputs as outliers. 1 = not outlier, -1 = outlier:")
print(pred_df.value_counts(normalize=True))


# add the prediction output as a new column into the df
df['output'] = predictions


# filter out the -1 (outliers) from the dataset
df_outliers_removed = df[df['output'] == 1]
print("The outliers have been removed from the dataset.")

#change pandas df back to dict
output_dict = df_outliers_removed.to_dict(orient='index')
print(f"The dataset now has {len(output_dict)} records.")

# output new dict to new pickle file 
## had to look this up: https://www.adamsmith.haus/python/answers/how-to-save-a-dictionary-to-a-file-with-pickle-in-python

file_to_write = open("output_dict.pkl", "wb")
pickle.dump(output_dict, file_to_write)
print("This script has completed without error. Please check folder for new output as pickle file.")
