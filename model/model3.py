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

    def train_test_split_data(self, data, test_size=0.4, random_state=42):
        if isinstance(data, pd.DataFrame):
            X = data.drop('target_column', axis=1)
            y = data['target_column']
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            raise ValueError("Data is not in the correct format for train-test split.")


class Models:
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

    def decision_tree_regression(self, X, Y, use_optuna=False):
        if use_optuna:
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
        else:
            self.model = DecisionTreeRegressor()

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def decision_tree_classifier(self, X, Y, use_optuna=False):
        if use_optuna:
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
        else:
            self.model = DecisionTreeClassifier()

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def random_forest_regression(self, X, Y, use_optuna=False):
        if use_optuna:
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
        else:
            self.model = RandomForestRegressor(random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def random_forest_classifier(self, X, Y, use_optuna=False):
        if use_optuna:
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
        else:
            self.model = RandomForestClassifier(random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)
## check if this is correct
    def gradient_boosting_regressor(self, X, Y, use_optuna=False):
        if use_optuna:
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
        else:
            self.model = GradientBoostingRegressor(random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)

    def gradient_boosting_classifier(self, X, Y, use_optuna=False):
        if use_optuna:
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
            self.model = GradientBoostingClassifier(**study.best_params, random_state=42)
        else:
            self.model = GradientBoostingClassifier(random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)
    
    def adaboost_classifier(self, X, Y, use_optuna=False):
        if use_optuna:
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
        else:
            self.model = AdaBoostClassifier(random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)
    
    def adaboost_regressor(self, X, Y, use_optuna=False):
        if use_optuna:
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
        else:
            self.model = AdaBoostRegressor(random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y)



    
