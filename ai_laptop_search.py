import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression as lreg
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
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

# Normalize numerical features
scaler = MinMaxScaler()
numerical_features_normalized = scaler.fit_transform(numerical_features)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=2024)
clusters = kmeans.fit_predict(numerical_features_normalized)

# Encode categorical features
encoder = OneHotEncoder(drop='first')
categorical_features = np.array([brand, processorBrand, processorTier, primaryStorageType,
                                 secondaryStorageType, gpuBrand, gpuType, isTouch, operatingSystem]).T
encoded_categorical_features = encoder.fit_transform(categorical_features)

# Convert encoded categorical features to dense array
encoded_categorical_features_dense = encoded_categorical_features.toarray()

# Combine normalized numerical and encoded categorical features
X = np.concatenate((numerical_features_normalized, encoded_categorical_features_dense), axis=1)

# Target variable
y = rating = np.array(rating)

# Train the linear regression model
regression_model = lreg()
regression_model.fit(X, y)

# Function to predict the AI rating of a laptop based on the specified features
def predict_laptop_rating(features):
    # Reshape the features array to match the input format of the model
    laptop_features = np.array(features).reshape(1, -1)

    # Predict the AI rating of the laptop
    predicted_rating = regression_model.predict(laptop_features)

    return predicted_rating[0]

# Function to recommend multiple laptops within the specified price limit, with the minimum desired rating,
# matching the preferred primary storage size, with the preferred RAM size, and with the preferred display size
def recommend_laptops_with_model(max_price_limit, preferred_min_ram_size, preferred_min_storage_size, preferred_display_size, cluster_indices, num_results=5):
    recommended_laptops = []
    
    # Iterate over laptops in the cluster
    for idx in cluster_indices:
        # Check if the price, storage size, RAM size, and display size meet the criteria
        if (max_price_limit is None or priceUSD[idx] <= max_price_limit) and \
           (preferred_storage_size is None or primaryStorageSize[idx] >= preferred_min_storage_size) and \
           (preferred_ram_size is None or ramSize[idx] >= preferred_min_ram_size) and \
           (preferred_display_size is None or displaySize[idx] == preferred_display_size):
            # Get the AI rating of the laptop
            ai_rating = predict_laptop_rating(X[idx])

            laptop_model = model[idx]  # Retrieve the model of the laptop
            actual_price = priceUSD[idx]  # Retrieve the actual price of the laptop
            recommended_laptops.append((laptop_model, ai_rating, actual_price))
    
    # Sort recommended laptops based on AI rating (descending order)
    recommended_laptops.sort(key=lambda x: x[1], reverse=True)
    
    # Filter the top num_results laptops within the price limit
    top_laptops_within_limit = []
    for laptop in recommended_laptops:
        if laptop[2] <= max_price_limit:
            top_laptops_within_limit.append(laptop)
            if len(top_laptops_within_limit) == num_results:
                break
    
    return top_laptops_within_limit

# Ask for the price
def prompt_price_limit():
    while True:
        try:
            price_limit = float(input("Enter your price limit: "))
            return price_limit
            break
        except ValueError:
            print("Please enter a valid price.")


# Function to prompt the user for their preferred RAM size
def prompt_preferred_ram_size():
    print("Do you have a preferred RAM size?")
    while True:
        response = input("Enter 'yes' or 'no': ").lower()
        if response == 'yes':
            # Define the list of available RAM sizes
            available_ram_sizes = [4, 8, 16, 32, 64]  # You can modify this list as needed
            
            # Prompt the user to select their preferred RAM size from the available options
            print("Available RAM sizes (in GB):", available_ram_sizes)
            while True:
                try:
                    preferred_ram_size = float(input("Enter your preferred RAM size in GB from the list above: "))
                    if preferred_ram_size not in available_ram_sizes:
                        raise ValueError("Please enter a RAM size from the available options.")
                    return preferred_ram_size
                except ValueError as e:
                    print(e)
            break
        elif response == 'no':
            return None
        else:
            print("Please enter 'yes' or 'no'.")

# Function to prompt the user for their preferred primary storage size
def prompt_preferred_storage_size():
    print("Do you have a preferred primary storage size?")
    while True:
        response = input("Enter 'yes' or 'no': ").lower()
        if response == 'yes':
            # Define the list of available primary storage sizes
            available_storage_sizes = [128, 256, 512, 1024, 2048]
            
            # Prompt the user to select their preferred primary storage size from the available options
            print("Available primary storage sizes (in GB):", available_storage_sizes)
            while True:
                try:
                    preferred_storage_size = float(input("Enter your preferred primary storage size in GB from the list above: "))
                    if preferred_storage_size not in available_storage_sizes:
                        raise ValueError("Please enter a storage size from the available options.")
                    return preferred_storage_size
                except ValueError as e:
                    print(e)
            break
        elif response == 'no':
            return None
        else:
            print("Please enter 'yes' or 'no'.")

# Function to prompt the user for their preferred display size
def prompt_preferred_display_size():
    print("Do you have a preferred display size?")
    while True:
        response = input("Enter 'yes' or 'no': ").lower()
        if response == 'yes':
            # Define the list of available display sizes
            available_display_sizes = [13.3, 14, 15.6, 17]  # You can modify this list as needed
            
            # Prompt the user to select their preferred display size from the available options
            print("Available display sizes (in inches):", available_display_sizes)
            while True:
                try:
                    preferred_display_size = float(input("Enter your preferred display size in inches from the list above: "))
                    if preferred_display_size not in available_display_sizes:
                        raise ValueError("Please enter a display size from the available options.")
                    return preferred_display_size
                except ValueError as e:
                    print(e)
            break
        elif response == 'no':
            return None
        else:
            print("Please enter 'yes' or 'no'.")

# Prompt the user for their price limit, preferred minimum rating, preferred RAM size, preferred primary storage size, and preferred display size
max_price_limit = prompt_price_limit()
preferred_ram_size = prompt_preferred_ram_size()
preferred_storage_size = prompt_preferred_storage_size()
preferred_display_size = prompt_preferred_display_size()

# Use k-means to find the cluster for the given user features
user_features = [max_price_limit, 0, 0, 0, preferred_ram_size, preferred_storage_size, 0, preferred_display_size, 0, 0]  # Placeholder value for missing features
if preferred_ram_size is None:
    user_features[4] = 0  # Placeholder value for missing preferred RAM size
if preferred_storage_size is None:
    user_features[5] = 0  # Placeholder value for missing preferred storage size
if preferred_display_size is None:
    user_features[7] = 0  # Placeholder value for missing preferred display size

users_cluster = kmeans.predict([user_features])[0]

# Find laptops within the cluster that match the user's preferences
cluster_indices = np.where(clusters == users_cluster)[0]

# Recommend multiple laptops meeting the criteria and retrieve their models and actual prices
recommended_laptops = recommend_laptops_with_model(max_price_limit, preferred_ram_size, preferred_storage_size, preferred_display_size, cluster_indices, 5)

if recommended_laptops:
    print("Recommended laptops within your price limit and other preferences")
    for i, laptop in enumerate(recommended_laptops, 1):
        print(f"Laptop {i}:")
        print("Model:", laptop[0])
        print("AI Rating:", laptop[1])  # Assuming this is the rating predicted by the AI model
        print("Price:", laptop[2])
        print()
else:
    print("Sorry, there are no laptops within your specified criteria.")