from model_factory.train_utils import TrainUtils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


class amz_linear_regression(object):
    """ This class will provide a linear regression model for the problem. """

    def __init__(self, model_path: str, ds_path: str = None):
        """
        The constructor for the amz_linear_regression class load model if exist or train new one.
        :param model_path: String, path for the model.
        """
        self.model = None
        if model_path is not None:
            self.model = load(model_path)
        elif ds_path is not None:
            self.model = self.train(ds_path)

    def train(self, ds_path: str):
        """ Train a linear regression model on the amazon ds. """
        train_utils = TrainUtils(ds_path)
        guided_df = train_utils.check_df(train_utils.df)
        print(guided_df)

        # pre-process work
        train_utils.handle_prices()
        train_utils.move_to_categorical(guided_df)

        # get train and test sets
        X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_utils.get_train_test_split()

        # Train the linear regression model
        linear_reg_enhanced = LinearRegression()
        linear_reg_enhanced.fit(X_train_comb, y_train_comb)

        # Predict on the testing set with enhanced features
        y_pred_linear_enhanced = linear_reg_enhanced.predict(X_test_comb)

        # Calculate the Root Mean Squared Error (RMSE) for the enhanced Linear Regression model
        rmse_linear_enhanced = mean_squared_error(y_test_comb, y_pred_linear_enhanced, squared=False)
        print(f"RMSE: {rmse_linear_enhanced}")

        # Save the model to a file
        model_filename = '../models/amazon_linear_reg.joblib'
        dump(linear_reg_enhanced, model_filename)

        return linear_reg_enhanced

    def show_linear_regression(self, model, X_test, y_test):

        # Assuming X_test is your test set features and y_test is the true target values
        # Replace 'model' with your loaded or defined model variable
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

        # Plot true vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line for reference
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        plt.show()

        # Optionally, plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals of Predictions')
        plt.show()


# sanity check
if __name__ == "__main__":
    # train
    # handler = amz_linear_regression(model_path=None, ds_path='../data/amazon_reviews.csv')

    # use
    handler = amz_linear_regression(model_path='../models/amazon_linear_reg.joblib')
    train_utils = TrainUtils(ds_path='../data/amazon_reviews.csv')
    guided_df = train_utils.check_df(train_utils.df)
    # pre-process work
    train_utils.handle_prices()
    train_utils.move_to_categorical(guided_df)
    X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_utils.get_train_test_split()
    handler.show_linear_regression(handler.model, X_test_comb, y_test_comb)










