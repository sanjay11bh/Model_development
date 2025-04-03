import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score , r2_score , f1_score ,mean_absolute_error , mean_squared_error
from sklearn.linear_model import LinearRegression , LogisticRegression , Lasso , SGDClassifier , SGDRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier , AdaBoostClassifier , AdaBoostRegressor,BaggingClassifier , BaggingRegressor ,GradientBoostingClassifier , GradientBoostingRegressor 
from sklearn.preprocessing import LabelEncoder

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

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

    def train_test_split_data(self, data, test_size=0.2, random_state=42):
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

    def linear_regressor(self ,X , Y):
        self.model = LinearRegression()
        self.model.fit(X,Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse
    """
    def logistic_regression(self ,X , Y):
        self.model = LogisticRegression()
        self.model.fit(X, Y)
        self.predictions = self.model.predict(X)
        r2 = r2_score(Y, self.predictions)
        mae = mean_absolute_error(Y, self.predictions)
        mse = mean_squared_error(Y, self.predictions)
        rmse = np.sqrt(mse)
        return self.model, self.predictions, r2, mae, mse, rmse"
        """
    
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
        criterion =  "squared_error"
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



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn.utils import shuffle
from sklearn.utils.multiclass import type_of_target

class check_best_model:
    def __init__(self, model_class, X_train, y_train):
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        
        # Determine whether the task is classification or regression
        self.task_type = 'classification' if type_of_target(y_train) in ['multiclass', 'binary'] else 'regression'

    def optimize(self, model_name, param_bounds, n_iter=10):
        def objective_function(**params):
            # Instantiate and train the model based on task type
            model = None
            if self.task_type == 'regression':
                if model_name == 'linear_regressor':
                    model, _, _, _, _, _ = self.model_class.linear_regressor(self.X_train, self.y_train)
                elif model_name == 'decision_tree_regression':
                    model, _, _, _, _, _ = self.model_class.decision_tree_regression(self.X_train, self.y_train)
                elif model_name == 'random_forest_regression':
                    model, _, _, _, _, _ = self.model_class.random_forest_regression(self.X_train, self.y_train)
                else:
                    raise ValueError(f"Unsupported regression model: {model_name}")

                # Use cross-validation to evaluate performance
                score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error').mean()

            elif self.task_type == 'classification':
                if model_name == 'random_forest_classifier':
                    model, _, _, _, _ = self.model_class.random_forest_classifier(self.X_train, self.y_train)
                elif model_name == 'decision_tree_classifier':
                    model, _, _, _, _ = self.model_class.decision_tree_classifier(self.X_train, self.y_train)
                else:
                    raise ValueError(f"Unsupported classification model: {model_name}")

                # Use cross-validation to evaluate accuracy
                score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy').mean()

            return score

        # Perform Bayesian Optimization
        optimizer = BayesianOptimization(f=objective_function, pbounds=param_bounds, random_state=42)
        optimizer.maximize(init_points=5, n_iter=n_iter)

        return optimizer.max

    def evaluate(self, model_type , params, X, y):
        print("model_type", model_type)
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        #score = model_type

        # Instantiate models based on task type and model type
        if "Linear Regression" in model_type:
            model = LinearRegression()
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = r2_score(y_val, predictions)

        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = accuracy_score(y_val, predictions)

        elif model_type == "Decision Tree Regression":
            model = DecisionTreeRegressor()
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = r2_score(y_val, predictions)

        elif model_type == "Random Forest Regression":
            model = RandomForestRegressor()
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = r2_score(y_val, predictions)

        elif model_type == "Random Forest Classifier":
            model = RandomForestClassifier()
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = accuracy_score(y_val, predictions)

        elif model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
            model.set_params(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = accuracy_score(y_val, predictions)

        return score
