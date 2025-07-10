import optuna
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score , r2_score , f1_score ,mean_absolute_error , mean_squared_error
from sklearn.linear_model import LinearRegression , LogisticRegression , Lasso , SGDClassifier , SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score


class ReadingData:
    def __init__(self):
        pass

    def csv_file(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def xlsx_file(self, file_path):
        data = pd.read_excel(file_path)
        return data

    def text_file(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        return data

    def train_test_split_data(self, data, test_size=0.3, random_state=42):
        if isinstance(data, pd.DataFrame):
            X = data.drop('target_column', axis=1) 
            y = data['target_column'] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X_train, X_test, y_train, y_test
        else:
            raise ValueError("Data is not in the correct format for train-test split.")
        

class Models:
    def __init__(self):
        self.model = None
        self.predictions = None

    def linear_regressor(self, X, Y):
        self.model = LinearRegression()
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse

    def logistic_regression(self, X, Y):
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1

    def decision_tree_regression(self, X, Y, use_optuna=False):
        if use_optuna:
            def objective(trial):
                max_depth = trial.suggest_int("max_depth", 2, 20)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                model.fit(X, Y)
                preds = model.predict(X)
                return mean_squared_error(Y, preds)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_params

            self.model = DecisionTreeRegressor(**best_params)
        else:
            self.model = DecisionTreeRegressor()

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse

    def lasso_regression(self, X, Y):
        self.model = Lasso(alpha=1.0)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse

    def decision_tree_classifier(self, X, Y, use_optuna=False):
        if use_optuna:
            def objective(trial):
                max_depth = trial.suggest_int("max_depth", 2, 20)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                model.fit(X, Y)
                preds = model.predict(X)
                return accuracy_score(Y, preds)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_params

            self.model = DecisionTreeClassifier(**best_params)
        else:
            self.model = DecisionTreeClassifier()

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1

    def random_forest_regression(self, X, Y, use_optuna=False):
        if use_optuna:
            def objective(trial):
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                max_depth = trial.suggest_int("max_depth", 2, 20)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
                criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])

                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                              criterion=criterion, random_state=42)
                model.fit(X, Y)
                preds = model.predict(X)
                return mean_squared_error(Y, preds)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_params

            self.model = RandomForestRegressor(**best_params, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42,
                                               criterion="squared_error", min_samples_split=2, min_samples_leaf=1)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse

    def random_forest_classifier(self, X, Y, use_optuna=False):
        if use_optuna:
            def objective(trial):
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                max_depth = trial.suggest_int("max_depth", 2, 20)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
                criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                               criterion=criterion, random_state=42)
                model.fit(X, Y)
                preds = model.predict(X)
                return accuracy_score(Y, preds)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_params

            self.model = RandomForestClassifier(**best_params, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42,
                                               criterion="gini", min_samples_split=2, min_samples_leaf=1)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1

    def gradient_boosting_classifier(self, X, Y, use_optuna=False):
        if use_optuna:
            def objective(trial):
                n_estimators = trial.suggest_int("n_estimators", 50, 200)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
                max_depth = trial.suggest_int("max_depth", 2, 10)

                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, 
                                                   max_depth=max_depth, random_state=42)
                model.fit(X, Y)
                preds = model.predict(X)
                return accuracy_score(Y, preds)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_params

            self.model = GradientBoostingClassifier(**best_params, random_state=42)
        else:
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1

    def gradient_boosting_regressor(self, X, Y, use_optuna=False):
        if use_optuna:
            def objective(trial):
                n_estimators = trial.suggest_int("n_estimators", 50, 200)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
                max_depth = trial.suggest_int("max_depth", 2, 10)

                model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, 
                                                  max_depth=max_depth, random_state=42)
                model.fit(X, Y)
                preds = model.predict(X)
                return mean_squared_error(Y, preds)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_params

            self.model = GradientBoostingRegressor(**best_params, random_state=42)
        else:
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse
