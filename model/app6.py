from model6 import ReadingData, Models , optuna_Model , AutoModelSelector
from sklearn.model_selection import train_test_split 
import os

def main() :
    rd = ReadingData()
    manual = Models()
    optuna = optuna_Model()
    auto = AutoModelSelector()

    # reading the data 
    File_path  = input("Enter the file path .csv , .xlsx , .txt : ")
    data = rd.read_data(File_path)

        #split in train test split
    ''' with open('columns_list.txt', 'w') as f:
        for col in data.columns:
            f.write(col + '\n')

    print("All column names saved to columns_list.txt")'''

    target = input("Enter the target column: ")
    if target in data.columns:
        X = data.drop(columns=[target])
        y = data[target]
    else:
        raise ValueError("Column 'target' not found in the dataset.")
    test_size = float(input("Enter the test size from 0 to 0.5 : "))
    X_train , X_test , y_train,y_test = train_test_split(X , y , test_size = test_size , random_state=42)
    #manual model

    print("Select a model:"
    "Enter 1 : for Manual Running "
    "Enter 2 : For optuna tunning and running"
    "Enter 3 for automated proccess ")

    choice = input("Enter the value 1, 2 and 3 :")

    if choice == '1' :
        print("Manual Model")
        task = input("for Regression press 1 else press any key : ")

        if task == '1':
            print("Enter 1 : For Linear Regression"
            "Enter 2 : For Random Forest"
            "Enter 3 : For Gradient Bossting ")
            choice = input("Enter the value 1 , 2 and 3 : ")

            if choice == '1':
                model, predictions, r2, mae, mse, rmse = manual.Linear_regressor(X_train, X_test, y_train, y_test)

            elif choice == '2' :
               model, predictions, r2, mae, mse, rmse = manual.Random_forest_regression(X_train, X_test, y_train, y_test)
            else :
                model, predictions, r2, mae, mse, rmse  = manual.AdaBoost_regressor(X_train, X_test, y_train, y_test)

            manual.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)
            manual.save_predictions_to_csv(y_test, predictions)
            manual.k_fold_cross_validation( model, X, y, n_splits=5, is_classification=False  )

        else :
            print("Enter 1 : For Linear Regression"
            "Enter 2 : For Random Forest"
            "Enter 3 : For Gradient Bossting ")
            choice = input("Enter the value 1 , 2 and 3 : ")
            if choice == '1' :
                model, pred, acc, f1 = manual.Logistic_regression(X_train, X_test, y_train, y_test)
            elif choice == '2' :
                model, pred, acc, f1 = manual.Random_forest_classifier(X_train, X_test, y_train, y_test)
            else :
                model, pred, acc, f1 = manual.AdaBoost_classifier(X_train, X_test, y_train, y_test)

            manual.save_predictions_to_csv(y_test, predictions)
            manual.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)
            manual.k_fold_cross_validation( model, X, y, n_splits=5, is_classification=True)


    elif choice == '2' :
        print("optuna Model")
        task = input("for Regression press 1 else press any key : ")

        if task == '1':
            print("Enter 1 : For Dicision Tree Regression"
            "Enter 2 : For Random Forest"
            "Enter 3 : For Ada Bossting ")
            choice = input("Enter the value 1 , 2 and 3 : ")

            if choice == '1':
                model, predictions, r2, mae, mse, rmse = optuna.Decision_tree_regression(X_train, X_test, y_train, y_test)

            elif choice == '2' :
               model, predictions, r2, mae, mse, rmse = optuna.Random_forest_regression(X_train, X_test, y_train, y_test)
            else :
                model, predictions, r2, mae, mse, rmse = manual.AdaBoost_regressor(X_train, X_test, y_train, y_test)

            optuna.save_predictions_to_csv(y_test, predictions)
            optuna.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)
            # optuna.optuna_visualization()
            print(f"R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")


        else :
            print("Enter 1 : For LOgistic Regression"
            "Enter 2 : For Random Forest"
            "Enter 3 : For ADA Bossting ")
            choice = input("Enter the value 1 , 2 and 3 : ")
            if choice == '1' :
                model, pred, acc, f1 = optuna.Logistic_regression(X_train, X_test, y_train, y_test)
            elif choice == '2' :
                model, pred, acc, f1 = optuna.Random_forest_classifier(X_train, X_test, y_train, y_test)
            else :
                model, pred, acc, f1 = optuna.AdaBoost_classifier(X_train, X_test, y_train, y_test)

            optuna.save_predictions_to_csv(y_test, predictions)
            optuna.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)
           ## optuna.optuna_visualization()
            print(f"f1: {f1:.4f}, acuuracy: {acc:.4f}")  
    
    elif choice == '3' :
        print("Atomated model")
        choice = input("Enter 1 for Regression and else for classification press any")

        if choice == '1':
            auto.run_regression(X_train, y_train, X_test, y_test) 

        else :
            auto.run_classification(X_train, y_train, X_test, y_test)




        
            


        
            


        


                

                












if __name__ == "__main__":  
    main()