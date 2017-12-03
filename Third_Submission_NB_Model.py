
# coding: utf-8

# In[1]:


# Since the full_train and full_test datasets were very big in size exceeding 16 gb in total
# We made use of cloud computing through using AWS S3 and EC2 :
# Datasets were uploaded on AWS S3 bucket
# Using an EC2 instance with 256 GB Ram we have used the following code :

# Import boto3 and io libraries to connect to Amazon S3 
import boto3
import io
import pandas as pd
import numpy as np


# Connect to the AWS S3 service
s3_client = boto3.client('s3')


# Connect to the S3 bucket and select the file to import
response = s3_client.get_object(Bucket="www.g43865368.gwu.edu",Key="full_train.csv")


# Save the full_train file as a Pandas Dataframe 
file = response["Body"].read()
train=pd.read_csv(io.BytesIO(file), delimiter=",")


# Free some memory by deleting unused variables
del response
del file


# Fill any Nan values with 0
train=train.fillna(0)


# Assign Features and Target Columns to variables 'features' and 'target' to be used in training the model 
features = list(train.values[:,4:])
target = list(train.values[:,3])


# Fit a Naive Bayes Classifyer Algorithm

# Import the Gaussian Navie Bayes classifier from the sklearn library
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# Train the Naive Bayes model according to training dataset Features and Targets
clf.fit(features, target)


# Free some memory deleting the train dataset as it won't be used anymore 
del train


#Import the test dataset that will be used for submission of the predictions

# Connect to the AWS S3 service
s3_client = boto3.client('s3')

# Connect to the S3 bucket and select the file to import
response = s3_client.get_object(Bucket="www.g43865368.gwu.edu",Key="full_test.csv")

# Save the full_test file as a Pandas Dataframe 
file = response["Body"].read()
test=pd.read_csv(io.BytesIO(file), delimiter=",")


# Fill Nan values with 0
test=test.fillna(0)


# Assign Features Columns from Test dataset to variables 'features' to be used in predicting the targets 
features = list(test.values[:,4:])


# Predict Target Probability for the test 
target_pred = clf.predict_proba(features)


# Create a for loop to predict each row of the final test and save it to final_pred dataframe
final_pred=[]
for i in (list(range(len(test)))):
    #print(i)
    test_id = str(test.id.iloc[i])
    
    # Because the predict_proba gives as array for the probability for each class 0 and 1 in our case. We will only use the
    # Probability of class 1 which is the second element of the array
    
    predicted_rating = target_pred[i][1]

    final_pred.append([test_id,predicted_rating])
    
final_pred=pd.DataFrame(final_pred)
final_pred.columns=['id','target']

# Save the final_pred to CSV to submit
final_pred.to_csv('third_submission.csv',index=False,float_format = '%.5f')

