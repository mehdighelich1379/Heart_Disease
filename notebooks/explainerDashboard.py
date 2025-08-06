import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import   MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

def make_pipeline(data_path, base_model):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    prep = ColumnTransformer(
        transformers=[
            ('scale', MinMaxScaler(), numeric_features)
        ],
        remainder='passthrough'  
    )

    pipeline = Pipeline(steps=[
        ('preprocessing', prep),
        ('model', base_model)
    ])

    return pipeline, X, y

model = RandomForestClassifier(n_estimators=350 , max_depth=10 , n_jobs=-1 , min_samples_split=2 , max_features=None , min_samples_leaf=2)

pipeline , X , y = make_pipeline(data_path="datasets/heart.csv" , base_model=model)



x_train , x_test , y_train , y_test = train_test_split(X , y , random_state=42  ,  test_size=0.2)
pipeline.fit(x_train , y_train)
joblib.dump(value=pipeline , filename='pipeline_heart.pkl')

from explainerdashboard import ExplainerDashboard , ClassifierExplainer

expinder = ClassifierExplainer(pipeline ,x_test , y_test )
ExplainerDashboard(expinder).run()
