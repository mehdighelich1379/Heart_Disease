# preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """ایجاد ویژگی‌های جدید از داده‌های ورودی"""
    data = data.copy()

    # ویژگی‌های ترکیبی
    data['age_chol_ratio'] = data['chol'] / (data['age'] + 1)
    data['bp_chol_product'] = data['trestbps'] * data['chol']
    data['restecg_thalach_diff'] = data['thalach'] - data['trestbps']

    # باینری‌سازی
    data['high_chol_flag'] = (data['chol'] > 240).astype(int)
    data['high_bp_flag'] = (data['trestbps'] > 130).astype(int)
    data['low_thalach_flag'] = (data['thalach'] < 120).astype(int)

    # تعامل ویژگی‌ها
    data['age_exercise_risk'] = data['age'] * data['exang']
    data['st_slope_risk'] = data['oldpeak'] * data['slope']

    return data

def preprocessing_pipeline(data: pd.DataFrame, target_col="target", base_model=LogisticRegression()):
    """ساخت پایپ‌لاین که شامل Feature Engineering و Scaling باشد"""
    # حذف داده‌های تکراری
    data = data.drop_duplicates()

    # جدا کردن X و y
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # مرحله Feature Engineering داخل Pipeline
    feat_eng_step = FunctionTransformer(feature_engineering, validate=False)

    # نام ستون‌ها بعد از feature engineering
    engineered_columns = feature_engineering(X.copy()).columns.tolist()

    # مرحله Scaling
    prep = ColumnTransformer(transformers=[
        ('scale', StandardScaler(), engineered_columns)
    ])

    # ساخت پایپ‌لاین
    pipeline = Pipeline(steps=[
        ('feature_engineering', feat_eng_step),
        ('preprocessing', prep),
        ('model', base_model)
    ])

    return pipeline, X, y
