import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_data(descriptor_file, target_column):
    df = pd.read_excel(descriptor_file)
    X = df.drop(columns=[target_column]).select_dtypes(include=[float, int])
    y = df[target_column]
    X = X[~y.isna()]
    y = y.dropna()
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X, y

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def shap_analysis(model, X, output_image):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.savefig(output_image)
    print(f"SHAP importance plot has been saved to {output_image}")

    # Calculate and sort SHAP values by their mean absolute values
    feature_importances = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)

    # Print top 24 features based on SHAP values
    top_features = feature_importances.head(24)
    print("Top 24 feature importances based on SHAP values:")
    print(top_features)
    plt.show()

def main(descriptor_file, target_column, output_image):
    X, y = load_data(descriptor_file, target_column)
    if X.empty:
        print("No numeric data available for model training.")
        return
    model = train_model(X, y)
    shap_analysis(model, X, output_image)

# Usage example
descriptor_file = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\descriptors2410.xlsx'
target_column = 'Output'
output_image = 'shap_summary_plot.png'

main(descriptor_file, target_column, output_image)
