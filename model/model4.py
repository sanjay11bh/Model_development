import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, r2_score, f1_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingClassifier, GradientBoostingRegressor , 
    AdaBoostClassifier , AdaBoostRegressor 
)


class ReadingData:
    def __init__(self):
        pass

    def csv_file(self, file_path):
        return pd.read_csv(file_path)

    def xlsx_file(self, file_path):
        return pd.read_excel(file_path)

    def text_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def train_test_split_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)


class Models:
    def __init__(self):
        self.model = None
        self.predictions = None

    def get_param(self, prompt, default, cast_func):
        val = input(f"{prompt} (default={default}): ")
        return cast_func(val) if val else default

    def evaluate_regression(self, X, Y):
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse

    def evaluate_classification(self, X, Y):
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1

    def Linear_regressor(self, X, Y):
        self.model = LinearRegression()
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def Logistic_regression(self, X, Y):
        max_iter = self.get_param("Enter max_iter", 1000, int)
        self.model = LogisticRegression(max_iter=max_iter)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def Lasso_regression(self, X, Y):
        alpha = self.get_param("Enter alpha", 1.0, float)
        self.model = Lasso(alpha=alpha)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def Decision_tree_regression(self, X, Y):
        max_depth = self.get_param("Enter max_depth", None, int) if input("Set max_depth? (y/n): ").lower() == "y" else None
        min_samples_split = self.get_param("Enter min_samples_split", 2, int)
        min_samples_leaf = self.get_param("Enter min_samples_leaf", 1, int)
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def Decision_tree_classifier(self, X, Y):
        max_depth = self.get_param("Enter max_depth", None, int) if input("Set max_depth? (y/n): ").lower() == "y" else None
        min_samples_split = self.get_param("Enter min_samples_split", 2, int)
        min_samples_leaf = self.get_param("Enter min_samples_leaf", 1, int)
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def Random_forest_regression(self, X, Y):
        n_estimators = self.get_param("Enter n_estimators", 100, int)
        max_depth = self.get_param("Enter max_depth", None, int) if input("Set max_depth? (y/n): ").lower() == "y" else None
        min_samples_split = self.get_param("Enter min_samples_split", 2, int)
        min_samples_leaf = self.get_param("Enter min_samples_leaf", 1, int)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def Random_forest_classifier(self, X, Y):
        n_estimators = self.get_param("Enter n_estimators", 100, int)
        max_depth = self.get_param("Enter max_depth", None, int) if input("Set max_depth? (y/n): ").lower() == "y" else None
        min_samples_split = self.get_param("Enter min_samples_split", 2, int)
        min_samples_leaf = self.get_param("Enter min_samples_leaf", 1, int)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def Gradient_boosting_regressor(self, X, Y):
        n_estimators = self.get_param("Enter n_estimators", 100, int)
        learning_rate = self.get_param("Enter learning_rate", 0.1, float)
        max_depth = self.get_param("Enter max_depth", 3, int)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def Gradient_boosting_classifier(self, X, Y):
        n_estimators = self.get_param("Enter n_estimators", 100, int)
        learning_rate = self.get_param("Enter learning_rate", 0.1, float)
        max_depth = self.get_param("Enter max_depth", 3, int)
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def AdaBoost_classifier(self, X, Y):
        n_estimators = self.get_param("Enter n_estimators", 50, int)
        learning_rate = self.get_param("Enter learning_rate", 1.0, float)
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def AdaBoost_regressor(self, X, Y):
        n_estimators = self.get_param("Enter n_estimators", 50, int)
        learning_rate = self.get_param("Enter learning_rate", 1.0, float)
        self.model = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

  
class optuna_Model:
    def __init__(self):
        self.model = None
        self.predictions = None

    def evaluate_regression(self, X, Y):
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse

    def evaluate_classification(self, X, Y):
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1

    def linear_regressor(self, X, Y):
        self.model = LinearRegression()
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def logistic_regression(self, X, Y):
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def lasso_regression(self, X, Y):
        self.model = Lasso(alpha=1.0)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def decision_tree_regression(self, X, Y):
        def objective(trial):
            model = DecisionTreeRegressor(
                max_depth=trial.suggest_int("max_depth", 2, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10)
            )
            model.fit(X, Y)
            return mean_squared_error(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        self.model = DecisionTreeRegressor(**study.best_params)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def decision_tree_classifier(self, X, Y):
        def objective(trial):
            model = DecisionTreeClassifier(
                max_depth=trial.suggest_int("max_depth", 2, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10)
            )
            model.fit(X, Y)
            return -accuracy_score(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        self.model = DecisionTreeClassifier(**study.best_params)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def random_forest_regression(self, X, Y):
        def objective(trial):
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                criterion=trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
                random_state=42
            )
            model.fit(X, Y)
            return mean_squared_error(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        self.model = RandomForestRegressor(**study.best_params, random_state=42)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def random_forest_classifier(self, X, Y):
        def objective(trial):
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                random_state=42
            )
            model.fit(X, Y)
            return -accuracy_score(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        self.model = RandomForestClassifier(**study.best_params, random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def gradient_boosting_regressor(self, X, Y, use_optuna=False):
        def objective(trial):
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                random_state=42
            )
            model.fit(X, Y)
            return mean_squared_error(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        self.model = GradientBoostingRegressor(**study.best_params, random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def gradient_boosting_classifier(self, X, Y):
        def objective(trial):
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                random_state=42
            )
            model.fit(X, Y)
            return -accuracy_score(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)
    
    def adaboost_classifier(self, X, Y):
        def objective(trial):
            model = AdaBoostClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                random_state=42
            )
            model.fit(X, Y)
            return -accuracy_score(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        self.model = AdaBoostClassifier(**study.best_params, random_state=42)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)
    
    def adaboost_regressor(self, X, Y):
        def objective(trial):
            model = AdaBoostRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                random_state=42
            )
            model.fit(X, Y)
            return mean_squared_error(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        self.model = AdaBoostRegressor(**study.best_params, random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)



    
