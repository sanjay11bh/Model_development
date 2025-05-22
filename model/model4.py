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
from sklearn.multioutput import MultiOutputRegressor , MultiOutputClassifier
import pickle


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
        self.model =LinearRegression()
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
    
    def predict_and_save(self, model, test_df, file_name="predictions.pkl"):
        """Predict on an unseen dataframe and pickle the results.
        Returns (result_df, file_path)."""
        preds = model.predict(test_df)

        # Attach predictions column(s)
        result_df = test_df.copy()
        if preds.ndim == 1:
            result_df["Predictions"] = preds
        else:
            for i in range(preds.shape[1]):
                result_df[f"Pred_{i}"] = preds[:, i]

        with open(file_name, "wb") as f:
            pickle.dump(result_df, f)

        return result_df, file_name

  
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
        study.optimize(objective, n_trials=5)
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
        study.optimize(objective, n_trials=5)
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
        study.optimize(objective, n_trials=5)
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
        study.optimize(objective, n_trials=5)
        self.model = RandomForestClassifier(**study.best_params, random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_classification(X, Y)

    def gradient_boosting_regressor(self, X, Y, use_optuna=False):
        def objective(trial):
            model = MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                random_state=42
            ))
            model.fit(X, Y)
            return mean_squared_error(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        self.model = MultiOutputRegressor(GradientBoostingRegressor(**study.best_params, random_state=42))

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
        study.optimize(objective, n_trials=5)

        self.model = GradientBoostingClassifier(**study.best_params, random_state=42)
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
            model = MultiOutputRegressor(AdaBoostRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                random_state=42
            ))
            model.fit(X, Y)
            return mean_squared_error(Y, model.predict(X))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        self.model = MultiOutputRegressor(AdaBoostRegressor(**study.best_params, random_state=42))
        optuna.visualization.plot_optimization_history(study)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        return self.evaluate_regression(X, Y) 
    
    def predict_and_save(self, model, test_df, file_name="predictions.pkl"):
        """Predict on an unseen dataframe and pickle the results.
        Returns (result_df, file_path)."""
        preds = model.predict(test_df)

        # Attach predictions column(s)
        result_df = test_df.copy()
        if preds.ndim == 1:
            result_df["Predictions"] = preds
        else:
            for i in range(preds.shape[1]):
                result_df[f"Pred_{i}"] = preds[:, i]

        with open(file_name, "wb") as f:
            pickle.dump(result_df, f)

        return result_df, file_name
    




# Automated_running:
class AutoModelSelector:
    def __init__(self, task="regression"):
        self.task = task
        self.model_class = optuna_Model()
        self.best_model_name = None
        self.best_score = -np.inf
        self.best_model = None
        self.all_models_scores = []  # ← NEW

    def select_best_model(self, X, Y):
        self.all_models_scores = []  # Reset before each run

        model_methods = {
            "regression": {
                "linear_regressor": self.model_class.linear_regressor,
                "lasso_regression": self.model_class.lasso_regression,
                "decision_tree_regression": self.model_class.decision_tree_regression,
                "random_forest_regression": self.model_class.random_forest_regression,
                "gradient_boosting_regressor": self.model_class.gradient_boosting_regressor,
                "adaboost_regressor": self.model_class.adaboost_regressor,
            },
            "classification": {
                "logistic_regression": self.model_class.logistic_regression,
                "decision_tree_classifier": self.model_class.decision_tree_classifier,
                "random_forest_classifier": self.model_class.random_forest_classifier,
                "gradient_boosting_classifier": self.model_class.gradient_boosting_classifier,
                "adaboost_classifier": self.model_class.adaboost_classifier,
            }
        }

        for name, method in model_methods[self.task].items():
            try:
                print(f"Training {name}...")
                result = method(X, Y)
                score = result[2] if self.task == "regression" else result[3]

                self.all_models_scores.append((name, score))  # ← Store all

                if score > self.best_score:
                    self.best_score = score
                    self.best_model_name = name
                    self.best_model = result[0]
            except Exception as e:
                print(f"{name} failed: {e}")
                self.all_models_scores.append((name, f"Failed: {e}"))  # ← Even failed ones
                continue

        return self.best_model_name, self.best_model, self.best_score
    
    def run_regression(self, X, y):
        self.task = "regression"
        return self.select_best_model(X, y)  

    def run_classification(self, X, y):
        self.task = "classification"
        return self.select_best_model(X, y)  
    



