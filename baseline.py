import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

train  = pd.read_csv('../titanic/data/raw/train.csv')

print(train.head())