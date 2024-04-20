import laptop_database as ldb
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split as ttspl
from sklearn.linear_model import LinearRegression as lreg
from sklearn.metrics import mean_squared_error as mserr

import numpy as np

# Combine numerical features into a single array
numerical_features = np.array([ldb.rating, ldb.coreCount, ldb.threadCount, ldb.ramSize, 
                               ldb.primaryStorageSize, ldb.secondaryStorageSize, ldb.displaySize, 
                               ldb.resolutionWidth, ldb.resolutionHeight, ldb.warrentyPeriod
                               ]).T

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(numerical_features)

# Add cluster labels to the dataset
clustered_data = np.column_stack((numerical_features, clusters))

# Add cluster labels as additional features
X = np.column_stack((numerical_features, clusters))

# Target variable
y = np.array(ldb.priceUSD)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = ttspl(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
regression_model = lreg()
regression_model.fit(X_train, y_train)

# Predict the prices on the testing set
y_pred = regression_model.predict(X_test)

# Evaluate the model
mse = mserr(y_test, y_pred)
#print("Mean Squared Error:", mse)

# Function to predict the price of a laptop based on the specified features
def predict_laptop_price(features):
    # Reshape the features array to match the input format of the model
    laptop_features = np.array(features).reshape(1, -1)

    # Predict the price of the laptop
    predicted_price = regression_model.predict(laptop_features)

    return predicted_price[0]

# Function to recommend a laptop within the specified price limit
def recommend_laptop(max_price_limit, features):
    # Predict the price of the laptop based on the specified features
    predicted_price = predict_laptop_price(features)

    if predicted_price <= max_price_limit:
        print("Recommended laptop within your price limit:")
        print("Predicted price:", predicted_price)
    else:
        print("Sorry, there are no laptops within your price limit.")

    return