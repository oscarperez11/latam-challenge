import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from typing import Tuple, Union, List


MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

FEATURES_COLS = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]

THRESHOLD_IN_MINUTES = 15


class DelayModel:

    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self._model = joblib.load(MODEL_PATH)
        else:
            self._model = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        features = features.reindex(columns=FEATURES_COLS, fill_value=0)

        if target_column:
            data = data.copy()
            if 'delay' not in data.columns:
                data['min_diff'] = data.apply(
                    lambda row: (
                        datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S') -
                        datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
                    ).total_seconds() / 60,
                    axis=1
                )
                data['delay'] = np.where(data['min_diff'] > THRESHOLD_IN_MINUTES, 1, 0)

            target = data[[target_column]]
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        target_series = target.iloc[:, 0]
        n_y0 = (target_series == 0).sum()
        n_y1 = (target_series == 1).sum()
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale
        )
        self._model.fit(features, target_series)
        joblib.dump(self._model, MODEL_PATH)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return [int(pred) for pred in self._model.predict(features)]
