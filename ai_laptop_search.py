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

# Function to recommend multiple laptops within the specified price limit and with the minimum desired rating
def recommend_laptops_with_model(max_price_limit, min_rating, cluster_indices, num_results=5):
    recommended_laptops = []
    num_found = 0
    
    # Iterate over laptops in the cluster
    for idx in cluster_indices:
        # Check if the rating meets the minimum requirement
        if rating[idx] >= min_rating:
            # Predict the price of the laptop based on the specified features
            predicted_price = predict_laptop_price(X[idx])

            # Check if the predicted price is within the limit
            if predicted_price <= max_price_limit:
                laptop_model = model[idx]  # Retrieve the model of the laptop
                actual_price = priceUSD[idx]  # Retrieve the actual price of the laptop
                recommended_laptops.append((laptop_model, predicted_price, actual_price))
                num_found += 1
                
                if num_found == num_results:
                    break

    return recommended_laptops

# Ask for the price
while True:
    try:
        price_limit = float(input("Enter your price limit: "))
        break
    except ValueError:
        print("Please enter a valid price.")

# Ask for the minimum desired rating
while True:
    try:
        min_rating = float(input("Enter the minimum desired rating: "))
        break
    except ValueError:
        print("Please enter a valid rating.")

# Use k-means to find the cluster for the given price and rating
price_rating_cluster = kmeans.predict([[price_limit, min_rating, 0, 0, 0, 0, 0, 0, 0, 0]])[0]

# Find laptops within the cluster with the closest price to the given limit
cluster_indices = np.where(clusters == price_rating_cluster)[0]

# Find the closest laptop within the cluster based on price
closest_laptop_index = cluster_indices[np.argmin(np.abs(priceUSD[cluster_indices] - price_limit))]
closest_laptop_features = X[closest_laptop_index]

# Recommend multiple laptops meeting the criteria and retrieve their models and actual prices
recommended_laptops = recommend_laptops_with_model(price_limit, min_rating, cluster_indices, 5)

if recommended_laptops:
    print("Recommended laptops within your price limit and with a rating over", min_rating, ":")
    for i, laptop in enumerate(recommended_laptops, 1):
        print(f"Laptop {i}:")
        print("Model:", laptop[0])
        print("Predicted price:", laptop[1])
        print("Actual price:", laptop[2])
        print()
else:
    print("Sorry, there are no laptops within your price limit and with a rating over", min_rating, ".")