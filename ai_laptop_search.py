import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression as lreg
from sklearn.preprocessing import OneHotEncoder
from laptop_database import get_features_from_csv

# Combine numerical features into a single array
def combine_numerical_features(priceUSD, rating, coreCount, threadCount, ramSize, primaryStorageSize,
                               secondaryStorageSize, displaySize, resolutionWidth, resolutionHeight, warrentyPeriod):
    numerical_features = np.array([rating, coreCount, threadCount, ramSize, primaryStorageSize,
                                   secondaryStorageSize, displaySize, resolutionWidth, resolutionHeight,
                                   warrentyPeriod]).T
    return numerical_features

# Get features from CSV
filename = 'laptops.csv'
model, brand, processorBrand, processorTier, primaryStorageType, secondaryStorageType, gpuBrand, gpuType, \
isTouch, operatingSystem, priceUSD, rating, coreCount, threadCount, ramSize, primaryStorageSize, \
secondaryStorageSize, displaySize, resolutionWidth, resolutionHeight, warrentyPeriod = get_features_from_csv(filename)

# Combine numerical features
numerical_features = combine_numerical_features(priceUSD, rating, coreCount, threadCount, ramSize, primaryStorageSize,
                                                secondaryStorageSize, displaySize, resolutionWidth, resolutionHeight,
                                                warrentyPeriod)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(numerical_features)

# Encode categorical features
encoder = OneHotEncoder(drop='first')
categorical_features = np.array([brand, processorBrand, processorTier, primaryStorageType,
                                 secondaryStorageType, gpuBrand, gpuType, isTouch, operatingSystem]).T
encoded_categorical_features = encoder.fit_transform(categorical_features)

# Convert encoded categorical features to dense array
encoded_categorical_features_dense = encoded_categorical_features.toarray()

# Combine numerical and encoded categorical features
X = np.concatenate((numerical_features, encoded_categorical_features_dense), axis=1)

# Target variable
y = priceUSD = np.array(priceUSD)

# Train the linear regression model
regression_model = lreg()
regression_model.fit(X, y)

# Function to predict the price of a laptop based on the specified features
def predict_laptop_price(features):
    # Reshape the features array to match the input format of the model
    laptop_features = np.array(features).reshape(1, -1)

    # Predict the price of the laptop
    predicted_price = regression_model.predict(laptop_features)

    return predicted_price[0]

# Function to recommend a laptop within the specified price limit and return its model
def recommend_laptop_with_model(max_price_limit, features):
    # Predict the price of the laptop based on the specified features
    predicted_price = predict_laptop_price(features)

    if predicted_price <= max_price_limit:
        laptop_model = model[closest_laptop_index]  # Retrieve the model of the laptop
        return laptop_model, predicted_price
    else:
        return None, None

# Ask for the price
while True:
    try:
        price_limit = float(input("Enter your price limit: "))
        break
    except ValueError:
        print("Please enter a valid price.")

# Use k-means to find the cluster for the given price
price_cluster = kmeans.predict([[price_limit, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[0]

# Find a laptop within the cluster with the closest price to the given limit
cluster_indices = np.where(clusters == price_cluster)[0]
closest_laptop_index = cluster_indices[np.argmin(np.abs(priceUSD[cluster_indices] - price_limit))]
closest_laptop_features = X[closest_laptop_index]

# Recommend the closest laptop and retrieve its model
laptop_model, predicted_price = recommend_laptop_with_model(price_limit, closest_laptop_features)

if laptop_model is not None:
    print("Recommended laptop within your price limit:")
    print("Model:", laptop_model)
    print("Predicted price:", predicted_price)
else:
    print("Sorry, there are no laptops within your price limit.")