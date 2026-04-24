"""
客户流失预测模型
基于 Kaggle Telco 数据集的特征工程和建模
"""

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]
MULTI_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
DROP_COLS = ["customerID", "gender"]


class ChurnPredictor:
    """客户流失预测器"""

    def __init__(self, model_type: str = "random_forest", **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def prepare_features(
        self, df: pd.DataFrame, target_col: str = "Churn"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """特征工程：编码类别变量、处理缺失值"""
        logger.info("开始特征工程...")
        work = df.drop(columns=DROP_COLS + [target_col], errors="ignore")

        mask = work["TotalCharges"].isna()
        work.loc[mask, "TotalCharges"] = work.loc[mask, "tenure"] * work.loc[mask, "MonthlyCharges"]

        for col in BINARY_COLS:
            if col in work.columns:
                work[col] = (work[col] == "Yes").astype(int)

        work = pd.get_dummies(work, columns=MULTI_COLS, drop_first=True)

        self.feature_names = work.columns.tolist()
        y = df[target_col]

        logger.info(f"特征数: {len(self.feature_names)}, 样本数: {len(y)}")
        logger.info(f"流失分布: {y.value_counts().to_dict()}")
        return work, y

    def train(
        self, X: pd.DataFrame, y: pd.Series,
        test_size: float = 0.2, random_state: int = 42,
    ) -> Dict:
        """训练模型并返回评估结果"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(**self.model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        self.model.fit(X_train_scaled, y_train)

        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        logger.info(f"训练集准确率: {train_score:.4f}, 测试集准确率: {test_score:.4f}")

        return {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "X_test": X_test_scaled,
            "y_test": y_test,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型尚未训练")
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("模型尚未训练")
        if self.model_type != "random_forest":
            return None
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
