import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

print("Task 1: feature selection using a classifier evaluator")

# Load the mushroom dataset from a CSV file
data = pd.read_csv('./mushroom.csv')

# Convert all categorical features into dummy/indicator variables (one-hot encoding)
dummies = pd.get_dummies(data)
# Assign the full dataset to 'x', and the 'edibility' column (target) to 'y'
x, y = pd.get_dummies(data), pd.get_dummies(data['edibility'])

# Print the shapes of the feature matrix (x) and target matrix (y)
print('x-shape:', x.shape)
print('y-shape:', y.shape)

# Initialize the SelectKBest feature selector with the chi-squared statistical test
skb = SelectKBest(chi2, k=5) # Select the top 5 features
# Fit the selector to the data and transform the dataset to select the top 5 features
skb.fit(x, y)
x_new = skb.transform(x)

# Print the shape of the dataset after selecting the top 5 features
print('x_new-shape:', x_new.shape)

# Identify the selected features and print their names
selected = [dummies.columns[i] for i in skb.get_support(indices=True)]
print("Selected features:", ", ".join(selected))

print("\nTask 2: feature selection using PCA")

# Print the original shape of the feature matrix (before PCA)
print("Original data shape:", x.shape)

# Apply PCA to reduce the dimensionality of the feature matrix
# Is applied to reduce the dataset to 5 principal components, which are linear combinations of the original features
# the goal of PCA is to capture the maximum amount of variance in the data with the fewest number of components
pca = PCA(n_components=5)  # Choose the number of components
X_pca = pca.fit_transform(x)

# Print the shape of the feature matrix after PCA
print("PCA data shape:", X_pca.shape)

# Identify the features that contribute the most to each of the 5 principal components
# For each of the 5 components, the feature that has the highest weight is selected
# These are the features that explains the most variance of the data
best_features = [pca.components_[i].argmax() for i in range(X_pca.shape[1])]
# Stores the k best features in a string
feature_names = [x.columns[best_features[i]] for i in range(X_pca.shape[1])]
print("Features in which gives max variance:", ", ".join(feature_names))

print("\nTask 3: compare the results of the two methods")

# Loop through different values of 'k' for SelectKBest (e.g., 5, 15, 25)
# Starts at 5, up to 36, with steps of 10
for i in range(5,36,10):
    # SelectKBest
    skb = SelectKBest(chi2, k=i)
    skb.fit(x, y)
    skb_res = skb.transform(x)
    selected = [dummies.columns[i] for i in skb.get_support(indices=True)]

    # PCA
    pca = PCA(n_components=i)
    X_pca = pca.fit_transform(x)
    best_features = [pca.components_[i].argmax() for i in range(X_pca.shape[1])]
    feature_names = [x.columns[best_features[i]] for i in range(X_pca.shape[1])]

    print(f"For k={i} we get {len(set(selected).intersection(feature_names))} overlapping features:\n", set(selected).intersection(feature_names), "\n")