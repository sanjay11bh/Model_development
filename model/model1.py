import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score , r2_score , f1_score ,mean_absolute_error , mean_squared_error
from sklearn.linear_model import LinearRegression , LogisticRegression , Lasso , SGDClassifier , SGDRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier , AdaBoostClassifier , AdaBoostRegressor,BaggingClassifier , BaggingRegressor ,GradientBoostingClassifier , GradientBoostingRegressor 
from sklearn.preprocessing import LabelEncoder

import optuna
from sklearn.model_selection import cross_val_score
import numpy as np

from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

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

    def linear_regressor(self ,X_train , y_train , X_test , y_test ):
        self.model = LinearRegression()
        self.model.fit(X_train,y_train)
        self.predictions = self.model.predict(X_test)
        r2 = r2_score(y_test, self.predictions)
        mae = mean_absolute_error(y_test , self.predictions)
        mse = mean_squared_error(y_test, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse
    
    def logistic_regression(self , X , Y):
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1


    def decision_tree_regression(self ,X , Y):
        self.model = DecisionTreeRegressor()
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse
    
    def lasso_regression(self , X, Y):
        self.model = Lasso(alpha=1.0,  
                           fit_intercept=True, precompute=False, 
                           copy_X=True, max_iter=1000, tol=0.0001, 
                           warm_start=False, positive=False, 
                           random_state=None, selection='cyclic')
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse
              
    def decision_tree_classifier(self ,X , Y):
        self.model = DecisionTreeClassifier()
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1
    
    
    def random_forest_regression(self, X , Y):
        n_estimators =  100 
        max_depth =  None
        random_state = 42
        criterion =  "squared_error"
        min_samples_split =  2
        min_samples_leaf =  1

        self.model = RandomForestRegressor(n_estimators=n_estimators , max_depth=max_depth,
                                           random_state=random_state , 
                                           criterion=criterion , 
                                           min_samples_split=min_samples_split , 
                                           min_samples_leaf=min_samples_leaf)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse
    
    def random_forest_classifier(self,X,Y):
        n_estimators = 100
        max_depth =  None
        random_state =  42
        criterion =  "gini"
        min_samples_split =  2
        min_samples_leaf =  1

        self.model = RandomForestClassifier(n_estimators=n_estimators , max_depth=max_depth,
                                           random_state=random_state , 
                                           criterion=criterion , 
                                           min_samples_split=min_samples_split , 
                                           min_samples_leaf=min_samples_leaf)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1

    
    def SGD_classifier(self , X , Y):
        self.model = SGDClassifier(loss='log_loss', penalty='l2', 
                                   alpha=0.0001, l1_ratio=0.15, 
                                   fit_intercept=True, max_iter=1000)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1
    
    def SGD_regressor(self ,X , Y ):
        self.model = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                                   max_iter=1000, tol=0.001, shuffle=True,  learning_rate='invscaling', eta0=0.01, 
                                   early_stopping=False, validation_fraction=0.1, n_iter_no_change=5)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse
    
    def adaboost_classifier(self, X, Y):
        self.model = AdaBoostClassifier(estimator=None, n_estimators=50,
                                         learning_rate=0.01, algorithm='deprecated', random_state=42)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1
    
    def adaboost_regressor(self , X, Y ):
        self.model = AdaBoostRegressor(n_estimators=100 , loss="square" , learning_rate=0.01 , random_state=42)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y , self.predictions)
        mae  =mean_absolute_error(Y , self.predictions)
        mse = mean_squared_error(Y , self.predictions)
        rmse = np.sqrt(mse)
        return self.model , self.predictions , r2 , mae , mse , rmse
    
    def bagging_classifier(self , X , Y ):
        self.model = BaggingClassifier(estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                                        bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1
        
    def bagging_regressor(self , X , Y ):
        self.model = BaggingRegressor(estimator=None , n_estimators=10,  max_samples=1.0, max_features=1.0, bootstrap=True,
                                       bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y , self.predictions)
        mae  =mean_absolute_error(Y , self.predictions)
        mse = mean_squared_error(Y , self.predictions)
        rmse = np.sqrt(mse)
        return self.model , self.predictions , r2 , mae , mse , rmse
    
    def gradient_boosting_classifier(self , X, Y):
        self.model =  GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                                  min_samples_split=2, 
                                                 max_depth=3 ,  warm_start=False,
                                                  validation_fraction=0.1, tol=0.0001, ccp_alpha=0.0)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        acc = accuracy_score(Y, self.predictions)
        f1 = f1_score(Y, self.predictions, average="weighted")
        return self.model, self.predictions, acc, f1
    
    def gradient_boosting_regressor(self , X , Y ):
        self.model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                                                min_samples_split=2, min_samples_leaf=1, max_features=None, alpha=0.9,
                                                 tol=0.0001, ccp_alpha=0.0)
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y , self.predictions)
        mae  =mean_absolute_error(Y , self.predictions)
        mse = mean_squared_error(Y , self.predictions)
        rmse = np.sqrt(mse)
        return self.model , self.predictions , r2 , mae , mse , rmse
    


class OptunaTuner:
    def __init__(self, model_builder, param_suggester, X, y, cv=5, scoring='r2', n_trials=30):
        """
        model_builder: function to build model using trial parameters
        param_suggester: function to suggest hyperparameters using trial object
        """
        self.model_builder = model_builder
        self.param_suggester = param_suggester
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.best_model = None
        self.study = None

    def _objective(self, trial):
        params = self.param_suggester(trial)
        model = self.model_builder(params)
        score = cross_val_score(model, self.X, self.y, scoring=self.scoring, cv=self.cv).mean()
        return score

    def optimize(self):
        direction = 'maximize' if self.scoring in ['r2', 'accuracy'] else 'minimize'
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(self._objective, n_trials=self.n_trials)
        best_params = self.study.best_params
        self.best_model = self.model_builder(best_params)
        self.best_model.fit(self.X, self.y)

    def get_best_model(self):
        return self.best_model

    def get_best_params(self):
        return self.study.best_params if self.study else None

    def get_best_score(self):
        return self.study.best_value if self.study else None

class ModelSelector:
    def __init__(self, task='regression'):
        self.task = task
        self.results = {}

    def evaluate(self, model_name, model_fn, X, y):
        model_obj, _, score, *_ = model_fn(X, y)
        self.results[model_name] = score


    def run(self, models_dict, X, y):
        for name, model in models_dict.items():
            self.evaluate(name, model, X, y)
        best_model_name = max(self.results, key=self.results.get)
        return best_model_name, self.results[best_model_name]







        


        

