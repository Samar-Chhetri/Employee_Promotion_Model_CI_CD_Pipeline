import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score ,recall_score, confusion_matrix


# Load dataset
train_df = pd.read_csv("data/train.csv")

train_df = train_df.rename(columns={'awards_won?': 'awards_won'})

# Data Partitioning
x = train_df.loc[:, train_df.columns!='is_promoted']
y = train_df['is_promoted']

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=42)

# Data Preprocessing
def data_preprocessing(df, train_data, test_data):

    nominal_cat_features = df.select_dtypes('object').columns.difference(['region', 'education'])
    ordinal_cat_feature = ['education']
    num_features = df.select_dtypes(exclude='object').columns.difference(['employee_id', 'is_promoted'])

    ordinal_pipeline = Pipeline([
        ('si', SimpleImputer(strategy='most_frequent')),
        ('oe', OrdinalEncoder(categories=[["Below Secondary", "Bachelor's", "Master's & above"]])),
        ('ss', StandardScaler(with_mean=False))])

    nominal_pipeline = Pipeline([
        ('si', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder()),
        ('ss', StandardScaler(with_mean=False))])

    num_pipeline = Pipeline([
        ('si', SimpleImputer(strategy='median')),
        ('ss', StandardScaler(with_mean=False))])

    transformer = ColumnTransformer([
        ('ordinal_trnf', ordinal_pipeline, ordinal_cat_feature),
        ('nominal_trnf', nominal_pipeline, nominal_cat_features),
        ('num_trnf', num_pipeline, num_features)])

    train_arr = transformer.fit_transform(train_data)
    test_arr = transformer.transform(test_data)

    return (train_arr, test_arr)

X_train_arr, X_test_arr = data_preprocessing(df=train_df, train_data=X_train, test_data=X_test)


# Model Training
model = LogisticRegression()
model.fit(X_train_arr, y_train)

# Prediction
pred = model.predict(X_test_arr)


# Performance metrics
print(f"\n Precision score: {round(precision_score(y_test, pred),2)}")

with open('metrics.txt', 'w') as out_metric:
    out_metric.write(f"\n Precision score: {round(precision_score(y_test, pred),2)}")
