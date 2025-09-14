import pyodbc as sql
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.decomposition import TruncatedSVD

def Get_Data():
    server_Name = 'DESKTOP-F3Q1PP2'
    Conn = sql.connect(f'DSN={server_Name};Trusted_Connection=yes;')
    data = Conn.cursor()
    data.execute('select * from [Practice].[dbo].[data]')
    dataset = pd.DataFrame.from_records(data.fetchall(),columns=[i[0] for i in data.description])
    return  dataset

def Data_Preprocess():
    Dataset = Get_Data()
    Numerical_Features = Dataset.drop(['id', 'diagnosis'], axis=1)
    Numerical_Features_name = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    Categorical_Feature_name = ['diagnosis']
    label = LabelEncoder()
    Y_Scale = label.fit_transform(Dataset[Categorical_Feature_name])
    Numeric_pipe = Pipeline([
        ('SI',KNNImputer(n_neighbors=3)),
        ('Scale',StandardScaler()),
    ])

    X_Scale = Numeric_pipe.fit_transform(Dataset[Numerical_Features_name])

    return X_Scale,Y_Scale, Categorical_Feature_name, Numerical_Features_name

def train_Model():
    X_Scale,Y_Scale, Categorical_Feature_name,Numerical_Features_name = Data_Preprocess()
    x_train, x_test, y_train, y_test = train_test_split(X_Scale,Y_Scale,test_size=0.2)
    model = LogisticRegression()
    Train_pipe = Pipeline([
        ('TSVD',TruncatedSVD(n_components=15,random_state=42)),
        ('LogisticReg',LogisticRegression())
    ])
    Train_pipe.fit(x_train,y_train)
    joblib.dump(Train_pipe, "cancer_detection.pkl")
    return  print("Model has been trained")

train_Model()