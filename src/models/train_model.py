# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.data.preprocess import preprocessing_pipeline  # مسیر را با پروژه خودت تطبیق بده
from src.metrics.evaluate import results  # برای نمایش کامل نتایج
from catboost import CatBoostClassifier


cat_model = CatBoostClassifier(iterations=300,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=2,
                              eval_metric='F1',
                              bagging_temperature = 0.2,
                              od_type='Iter',
                              metric_period = 75,
                              od_wait=100,
                              random_state=42)  
def train_model():
    # --- 1. Load dataset ---
    data = pd.read_csv("src/data/heart.csv")  # مسیر relative برای main.py


    # --- 2. Create pipeline and get processed features ---
    pipeline, X, y = preprocessing_pipeline(data, target_col="target" , base_model=cat_model)

    # --- 3. Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    # --- 4. Fit the model ---
    pipeline.fit(X_train, y_train)

    # --- 5. Predictions ---
    y_pred = pipeline.predict(X_test)

    # --- 6. Quick metrics ---
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # --- 7. Full evaluation ---
    results(pipeline, X_test, y_test, X_train)

    # --- 8. Save model ---
    joblib.dump(pipeline, "heart_pipeline_model.pkl")
    print("✅ Model saved as heart_pipeline_model.pkl")

