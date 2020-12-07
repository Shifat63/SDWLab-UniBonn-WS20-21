from sklearn import preprocessing
import numpy as np
import pandas as pd

# file location
url = "diabetes_data.csv"
# Importing the datasets
dataset = pd.read_csv(url)
# mark ? values as missing or NaN
dataset.replace(to_replace="[?]", value=np.nan, regex=True, inplace=True)
dataset = dataset.dropna()
df = pd.DataFrame(dataset)
names = ["Age","Gender","Polyuria","Polydipsia","sudden weight loss","weakness","Polyphagia",
         "Genital thrush","visual blurring","Itching","Irritability","delayed healing",
         "partial paresis","muscle stiffness","Alopecia", "Obesity", "class"]
df = df[names]
df["Obesity"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["Alopecia"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["muscle stiffness"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["partial paresis"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["delayed healing"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["Irritability"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["Itching"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["visual blurring"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["Genital thrush"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["Polyphagia"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["weakness"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["sudden weight loss"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["Polydipsia"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["Polyuria"].replace({'No': 0, 'Yes': 1}, inplace = True)
df["Gender"].replace({'Male': 0, 'Female': 1}, inplace = True)
df["Age"] = preprocessing.scale(df["Age"])
# writing pre-processed data to a csv file
df.to_csv('diabetes_data_preprocessed.csv', sep=',', index=False)
# print(df)

