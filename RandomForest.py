# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'C:\Users\Coolc\Desktop\Uni\Year 2\Sem1-Mod-Artificial Intelligence\Coursework\Coursework(x.1.22)\Solar_Flares_Dataset.csv')
#print (df.head())
sizes = df["Flare_type"].value_counts(sort = 1)
print (sizes) 

#--Now to predict the dependent variable (Flare type) based on independent variables
#--First prepare the data for RF


#--Drop irrelevant columns
df.drop(["C-Class Flares"], axis = 1, inplace = True)
df.drop(["M-class flares"], axis = 1, inplace = True)
df.drop(["X-class flares"], axis = 1, inplace = True)
df.drop(["Area_of_the_largest_spot"], axis = 1, inplace = True)
df.drop(["Did_region_become_historically_complex_on_this_pass"], axis = 1, inplace = True)
#print (df.head())

#--Deal with missing values
df = df.dropna()

#--Covert non numeric data to numeric
df.Flare_type[df.Flare_type == "N"] = 0
df.Flare_type[df.Flare_type == "C"] = 1
df.Flare_type[df.Flare_type == "M"] = 2
df.Flare_type[df.Flare_type == "X"] = 3

df.modified_Zurich_class[df.modified_Zurich_class == "A"]= 1
df.modified_Zurich_class[df.modified_Zurich_class == "B"]= 2
df.modified_Zurich_class[df.modified_Zurich_class == "C"]= 3
df.modified_Zurich_class[df.modified_Zurich_class == "D"]= 4
df.modified_Zurich_class[df.modified_Zurich_class == "E"]= 5
df.modified_Zurich_class[df.modified_Zurich_class == "F"]= 6
df.modified_Zurich_class[df.modified_Zurich_class == "H"]= 7

df.largest_spot_size[df.largest_spot_size == "X"] = 1
df.largest_spot_size[df.largest_spot_size == "R"] = 2
df.largest_spot_size[df.largest_spot_size == "S"] = 3
df.largest_spot_size[df.largest_spot_size == "A"] = 4
df.largest_spot_size[df.largest_spot_size == "H"] = 5
df.largest_spot_size[df.largest_spot_size == "K"] = 6

df.spot_distribution[df.spot_distribution == "X"] = 1
df.spot_distribution[df.spot_distribution == "O"] = 2
df.spot_distribution[df.spot_distribution == "I"] = 3
df.spot_distribution[df.spot_distribution == "C"] = 4
#print (df.head())

#--Define dependent variable
Y = df["Flare_type"].values
Y = Y.astype("int")

#--Define independent variables
X = df.drop(labels = ["Flare_type"], axis = 1)

#--Split data into train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 20)

model = RandomForestClassifier(n_estimators = 10, random_state = 30)
model.fit(X_train, Y_train)
prediction_test = model.predict(X_test)

sizes = df["Flare_type"].value_counts(sort = 1)
print("Accuracy = ", round(metrics.accuracy_score(Y_test, prediction_test), 2))
print(classification_report(Y_test, prediction_test))

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index = feature_list).sort_values(ascending = False)
print(feature_imp)
