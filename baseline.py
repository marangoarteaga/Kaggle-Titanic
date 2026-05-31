import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


#Load data
train_data  = pd.read_csv('../titanic/data/raw/train.csv')
test_data = pd.read_csv('../titanic/data/raw/test.csv')

#Features
features = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
]

X = train_data[features]
X_test = test_data[features]

y = train_data["Survived"]

#Convert categorical variables

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

#Align columns 

X, X_test = X.align(
    X_test,
    join='left',
    axis=1,
    fill_value=0
)

#Missing values

imputer = SimpleImputer(strategy='median')

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)

#Model

model = RandomForestClassifier(
    random_state=0
)

model.fit(X, y)
predictions = model.predict(X_test)

#Submission

submission = pd.DataFrame({
    'PassengerId':test_data['PassengerId'],
    'Suvived' : predictions
})

submission.to_csv(
    'submissions/submissions_01.csv',
    index=False
)
print('Submission created succesfully.')