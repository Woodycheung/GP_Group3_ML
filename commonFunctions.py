"""
author:
CHAN Chung Hang  22061759S
CHEUNG Ho Bun  22056983S
POON Wing Fung 22056100S
YEUNG Ka Wai 22049550S
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from sklearn import tree
from sklearn.tree import export_text, export_graphviz
import graphviz


# Redefine the function to determine the quarter based on the month
def get_quarter(month):
    if 1 <= month <= 3:
        return 1
    elif 4 <= month <= 6:
        return 2
    elif 7 <= month <= 9:
        return 3
    elif 10 <= month <= 12:
        return 4


def analysis_plot_result(merged_data, flag):
    # Analysis with all data from merged_data
    # Create a result_dict for store the model result
    results_dict = {}
    # Array for all country
    country_columns = ['Africa', 'Americas', 'Australia_NewZealand_SouthPacific', 'Europe',
                       'MiddleEast', 'NorthAsia', 'SouthAsia_SoutheastAsia', 'MainlandChina', 'Taiwan', 'Macau',
                       'Total']

    for country in country_columns:
        # plot the scatter results for raw data from Visitor and GDP Data in Merge Data
        plot_visitor_GDP_results_with_raw_data(merged_data, country, "Results", flag)

        X = merged_data[[country]].values.flatten()
        y = merged_data['GDP_Million_HKD'].values

        # Ensure X & y is numeric and handle any non-numeric entries
        X = pd.to_numeric(X, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')  # This will convert non-numeric values to NaN
        mask = ~np.isnan(X) & ~np.isnan(y)  # This will remove NaN values that were non-numeric
        X = X[mask]
        y = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the models
        # LinearRegression
        model_lr = LinearRegression()
        model_lr.fit(X_train.reshape(-1, 1), y_train)
        y_pred_lr = model_lr.predict(X_test.reshape(-1, 1))
        r2_lr = r2_score(y_test, y_pred_lr)
        # Pearson correlation coefficient for Linear Regression
        corr_coef_lr, _ = pearsonr(y_test, y_pred_lr)
        plot_linear_results(X_test, y_test, y_pred_lr, f"{flag} LinearRegression for {country}", "Linear")

        # RandomForest Regressor
        t = 30
        model_rf = RandomForestRegressor(n_estimators=t, max_depth=2)
        model_rf.fit(X_train.reshape(-1, 1), y_train)
        y_pred_rf = model_rf.predict(X_test.reshape(-1, 1))
        r2_rf = r2_score(y_test, y_pred_rf)

        # plot tree dot
        estimator = model_rf.estimators_[0]
        dot_filename = f"{flag}_{country}_tree.dot"
        graph_filename = f"{flag}_{country}_Source.gv"
        savePath = "RandomForest"
        export_graphviz(estimator, out_file=os.path.join(savePath, dot_filename),
                        feature_names=['number_of_visitors'],
                        class_names=['output'],
                        rounded=True, proportion=True,
                        precision=2, filled=True)
        with open(os.path.join(savePath, dot_filename)) as f:
            dot_graph = f.read()
        graph = graphviz.Source(dot_graph)
        graph.render(filename=os.path.join(savePath, graph_filename), format='pdf', view=True)

        # Polynomial Regression
        degree = 2
        model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model_poly.fit(X_train.reshape(-1, 1), y_train)
        y_pred_poly = model_poly.predict(X_test.reshape(-1, 1))
        r2_poly = r2_score(y_test, y_pred_poly)
        plot_poly_results(X_test, y_test, model_poly, degree,
                          f"{flag} Polynomial Regression of {country} - degree = {degree}", "Poly")

        # K-means cal
        # kmeans = KMeans(n_clusters=3, random_state=42)
        # clusters = kmeans.fit_predict(X.reshape(-1, 1))

        # plot scatter plot with jitter for clarity
        # jitter_strength = 0.05 * np.ptp(X)  # 5% of the range of X
        # x_jittered = X + np.random.normal(0, jitter_strength, size=X.shape)

        # log number change
        # y_log = np.log1p(y)

        # # Plot the scatter plot with jitter for clarity
        # plt.figure(figsize=(12, 8))
        # scatter = plt.scatter(x=x_jittered, y=y_log, c=clusters, cmap='viridis', alpha=0.6)
        # plt.colorbar(scatter, label='Cluster')
        # plt.title(f'{data_type} Scatter Plot of {country} Visitors vs GDP with Cluster Coloring (With K-Means Clustering)')
        # plt.xlabel(f'Number of Visitors from {country} (jittered)')
        # plt.ylabel('GDP in Million HKD')
        # plt.grid(True)
        # #plt.show()
        # plt.savefig(fr"K-Means/{country}{data_type}.png")
        # plt.close()

        results_dict[country] = {
            'Linear Regression': {'r2_score': r2_lr, 'pearson_r': corr_coef_lr},
            'Random Forest': {'r2_score': r2_rf},
            # 'KMeans': {'clusters': clusters},
            'Polynomial Regression': {'r2_score': r2_poly},
        }

    print_result(results_dict)

    # Draw heatmaps
    # plt.figure(figsize=(12, 10))
    # correlation_matrix = merged_data[country_columns + ['GDP_Million_HKD']].corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.title('Correlation Heatmap')
    # #plt.show()
    # plt.savefig(fr"heatmaps/{data_type}.png")
    # plt.close()


def plot_visitor_GDP_results_with_raw_data(merged_data, country, folder, flag):
    plt.figure(figsize=(10, 8))
    plt.scatter(merged_data[country], merged_data['GDP_Million_HKD'], alpha=0.5)
    plt.title(f'{flag} Relationship Between Number of Visitors and GDP for {country}')
    plt.xlabel('Number of Visitors')
    plt.ylabel('GDP (Million HKD)')
    # Format the y-axis ticks with larger interval and with thousand separator
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    # Rotate the y-axis labels
    plt.setp(ax.get_yticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout
    plt.savefig(f"{folder}/{flag} {country}_scatter.png")
    plt.close()


def plot_poly_results(X_test, y_test, model_poly, degree, title, folder):
    x_grid = np.linspace(X_test.min(), X_test.max(), 300).reshape(-1, 1)
    y_grid_pred = model_poly.predict(x_grid)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_test, y_test, color='blue', label='Test Data')
    plt.plot(x_grid, y_grid_pred, color='red', label=f'Polynomial Degree {degree} Regression Line')
    plt.title(f"{title}")
    plt.xlabel(f'Visitors')
    plt.ylabel('GDP (Million HKD)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"{folder}/{title}.png")
    plt.close()


def plot_linear_results(X_test, y_test, y_pred, title, folder):
    plt.figure(figsize=(10, 8))
    plt.scatter(X_test, y_test, color='blue', label='Test data')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')
    plt.xlabel(f'Visitor')
    plt.ylabel('Predicted GDP')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(fr"{folder}/{title}.png")
    plt.close()


def plot_GDP_results(y_test, y_pred, title, folder):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual GDP')
    plt.ylabel('Predicted GDP')
    plt.title(title)
    plt.grid(True)
    # plt.show()
    plt.savefig(fr"{folder}/{title}.png")
    plt.close()


# def plt_algorithm_result():


def print_result(analysis_results):
    for country, result in analysis_results.items():
        lr_r2 = result['Linear Regression']['r2_score']
        lr_pearson_r = result['Linear Regression'].get('pearson_r', None)  # Use .get() to avoid KeyError if not present
        rf_r2 = result['Random Forest']['r2_score']
        r2_poly = result['Polynomial Regression']['r2_score']

        print(f"Analysis for {country}:")
        if lr_r2 is not None:
            print(f"The Linear Regression model's R^2 score is {lr_r2:.2f}. This means that "
                  f"{lr_r2 * 100:.1f}% of the variance in GDP is explained by the number of visitors from {country}.")
        if lr_pearson_r is not None:
            print(f"The Linear Regression model's Pearson correlation coefficient is {lr_pearson_r:.2f} for {country}")
        if r2_poly is not None:
            print(f"The Polynomial Regression model's R^2 score is {r2_poly:.2f}. This means that "
                  f"{r2_poly * 100:.1f}% of the variance in GDP is explained by the number of visitors from {country}.")
        if rf_r2 is not None:
            if rf_r2 > lr_r2:
                accuracy_statement = "indicating a higher level of accuracy in prediction compared to the Linear Regression model."
            else:
                accuracy_statement = "indicating a similar or lower level of accuracy in prediction compared to the Linear Regression model."
            print(f"The Random Forest model's R^2 score is {rf_r2:.2f}. This means that "
                  f"{rf_r2 * 100:.1f}% of the variance in GDP is explained by the number of visitors from {country}, {accuracy_statement}")
        else:
            print("The Random Forest model's R^2 score is not available for this analysis.")
        print()

# No need to use cause data too small for cross optimize
# def compare_and_optimize_models(merged_data):
#     # Specify the country columns to analyze
#     country_columns = ['Africa', 'Americas', 'Australia_NewZealand_SouthPacific', 'Europe',
#                        'MiddleEast', 'NorthAsia', 'SouthAsia_SoutheastAsia', 'MainlandChina', 'Taiwan', 'Macau',
#                        'Total']
#     results = {}
#     for country in country_columns:
#         print(f"Analyzing data for {country}")
#         X = merged_data[[country]].values.flatten()
#         y = merged_data['GDP_Million_HKD'].values
#
#         # Ensure y is numeric and handle any non-numeric entries
#         y = pd.to_numeric(y, errors='coerce')  # Convert non-numeric values to NaN
#         X = X[~np.isnan(y)]  # Remove corresponding X entries
#         y = y[~np.isnan(y)]  # Remove NaN values from y
#
#         if len(y) < 5:  # Ensure enough data points for 5-fold cross-validation
#             print(f"Not enough data for {country} to perform 5-fold cross-validation.")
#             continue
#
#         # Models to evaluate
#         model_lr = LinearRegression()
#         model_rf = RandomForestRegressor(random_state=42)
#         model_poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
#
#         # Linear Regression
#         scores_lr = cross_val_score(model_lr, X.reshape(-1, 1), y, cv=5, scoring='r2')
#         print(f"Linear Regression - Average R² score: {scores_lr.mean():.3f}")
#
#         # Polynomial Regression
#         scores_poly = cross_val_score(model_poly, X.reshape(-1, 1), y, cv=5, scoring='r2')
#         print(f"Polynomial Regression - Average R² score: {scores_poly.mean():.3f}")
#
#         # Random Forest Regression with Grid Search
#         param_grid_rf = {
#             'n_estimators': [50, 100, 150],
#             'max_features': ['sqrt', 'log2'],
#             'max_depth': [None, 10, 20, 30]
#         }
#
#         grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='r2', error_score='raise')
#         try:
#             grid_search_rf.fit(X.reshape(-1, 1), y)
#             print(f"Random Forest - Best R² score: {grid_search_rf.best_score_:.3f}")
#             print(f"Best parameters: {grid_search_rf.best_params_}")
#         except ValueError as e:
#             print(f"Error Msg:{e}")
#
#         results[country] = {
#             'Linear Regression': scores_lr.mean(),
#             'Polynomial Regression': scores_poly.mean(),
#             'Random Forest': grid_search_rf.best_score_
#         }
#
#     print_result(results)
