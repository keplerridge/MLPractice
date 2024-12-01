#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os

#%%
df = pd.read_csv('./ml_ready.tsv', sep = '\t')
df.drop(columns = ['Unnamed: 0'])

#%%
def run_model(df, user, random_seed = 42, round_to = 6, features = 11794):
    """
    df: pandas dataframe with genetic data on breast cancer
    user: name of who is running the model
    random_seed = this will create reproducibility in selecting our random features
    round_to: the amount of decimal places to round to, default is 6 because that is what the data is already rounded to
    features: the amount of columns to use in the model, default is 11795 which is all of them

    returns: it doesn't return anything but it prints out the area under the ROC curve and appends or creates a new TSV file with the results
    """

    # Indentify class as my target column and all other columns as
    # the features to use and split the dataframe into 75/25 
    # training and test set
    target = 'Class'

    # Make sure we don't try to use too many features
    if features > len(df.columns) - 1:
        raise ValueError("The number of features requested exceeds available features.")

    # Get all of the feature columns
    feature_columns = df.drop(columns=target).columns.tolist()

    # Randomly sample the specified number of feature columns
    selected_features = pd.Series(feature_columns).sample(n=features, random_state=random_seed).tolist()


    X = df[selected_features]

    # Round all values in the dataframe
    X = X.round(round_to)

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 42)

    # Create model
    regression = LogisticRegression(random_state = 16, max_iter = 1000)

    # Fit the model
    regression.fit(X_train, y_train)

    # Make predictions
    class_prediction = regression.predict(X_test)

    # Make predictions
    prediction = regression.predict(X_test)

    # Evaluation of model
    predict_proba = regression.predict_proba(X_test)[:,1]

    # This gets the false positive rate, true positive rate, and the thresholds for that
    false_pos, true_pos, thresholds = metrics.roc_curve(y_test, predict_proba)

    auc = metrics.roc_auc_score(y_test, predict_proba)
    print(f"Area Under the ROC Curve (AUC): {auc:.4f}")

    new_data = pd.DataFrame({
        "User": [user],
        "Decimals rounded to": [round_to],
        "Random Seed" : [random_seed],
        "Number of features used": [features],
        "AUC": [auc]
    })

    file_path = './updated_results.tsv'

    # Check if the file is empty, if so we will keep the header otherwise we will not
    header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

    new_data.to_csv(file_path, mode = 'a', sep = '\t', index = False, header = header)

# %%
# This iterates through and runs the model 10 times with a new random seed
# for selecting features on each iteration
iterations = 1

while iterations < 11:
    run_model(df, 'Kepler', iterations, 6)
    iterations += 1

# %%
# Load the results from the models
results_df = pd.read_csv('./updated_results.tsv', sep='\t')

# Get the average for each set of features
average_results = results_df.groupby("Number of features used")["AUC"].mean().reset_index()

# Output the new df to a TSV
average_results.to_csv('./average_results.tsv', sep = '\t', index = False)

# %%
