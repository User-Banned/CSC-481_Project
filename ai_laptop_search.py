import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression as lreg
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from laptop_database import get_features_from_csv

# Combine numerical features into a single array
def combine_numerical_features(priceUSD, rating, coreCount, threadCount, ramSize, primaryStorageSize,
                               secondaryStorageSize, displaySize, resolutionWidth, resolutionHeight, warrentyPeriod):
    numerical_features = np.array([priceUSD, coreCount, threadCount, ramSize, primaryStorageSize,
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
n_clusters=6
kmeans = KMeans(n_clusters, random_state=2024)
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

# Initialize a dictionary to store cluster-specific models
cluster_models = {}

# Train linear regression models for each cluster
for cluster_idx in range(n_clusters):
    # Filter data points belonging to the current cluster
    cluster_data_indices = np.where(kmeans.labels_ == cluster_idx)[0]
    cluster_features = X[cluster_data_indices]
    cluster_ratings = y[cluster_data_indices]
    
    # Train linear regression model for the current cluster
    cluster_regression_model = lreg()
    cluster_regression_model.fit(cluster_features, cluster_ratings)
    
    # Store the trained model in the dictionary
    cluster_models[cluster_idx] = cluster_regression_model

# Modify prediction function to accept cluster index and use the corresponding model
def predict_laptop_rating(features, cluster_idx):
    # Retrieve the cluster-specific model
    cluster_model = cluster_models[cluster_idx]
    
    # Reshape the features array to match the input format of the model
    laptop_features = np.array(features).reshape(1, -1)

    # Predict the rating of the laptop using the cluster-specific model
    predicted_rating = cluster_model.predict(laptop_features)

    return predicted_rating[0]

# Function to recommend multiple laptops within the specified price limit, with the minimum desired rating,
# matching the preferred primary storage size, with the preferred RAM size, and with the preferred display size
def recommend_laptops_with_model(max_price_limit, preferred_min_ram_size, preferred_min_storage_size, preferred_display_size, cluster_indices, num_results=n_clusters):
    recommended_laptops = []
    
    # Iterate over laptops in the cluster
    for idx in cluster_indices:
        # Check if the price, storage size, RAM size, and display size meet the criteria
        if (max_price_limit is None or priceUSD[idx] <= max_price_limit) and \
           (preferred_storage_size is None or primaryStorageSize[idx] >= preferred_min_storage_size) and \
           (preferred_ram_size is None or ramSize[idx] >= preferred_min_ram_size) and \
           (preferred_display_size is None or displaySize[idx] == preferred_display_size):
            # Get the cluster index for the current laptop
            laptop_cluster_idx = kmeans.labels_[idx]
            
            # Predict the rating of the laptop using the cluster-specific model
            ai_rating = predict_laptop_rating(X[idx], laptop_cluster_idx)
            
            laptop_model = model[idx]  # Retrieve the model of the laptop
            laptop_price = priceUSD[idx]  # Retrieve the price of the laptop
            recommended_laptops.append((laptop_model, ai_rating, laptop_price))
    
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

# Function to recommend multiple laptops from each cluster
def recommend_laptops_from_clusters(max_price_limit, preferred_min_ram_size, preferred_min_storage_size, preferred_display_size, clusters, num_results_per_cluster=1):
    recommended_laptops = []
    
    # Iterate over each cluster
    for cluster_idx in range(clusters.n_clusters):
        # Find laptops within the cluster
        cluster_indices = np.where(clusters.labels_ == cluster_idx)[0]
        
        # Recommend laptops within the cluster using cluster-specific model
        cluster_recommendations = recommend_laptops_with_model(max_price_limit, preferred_min_ram_size, preferred_min_storage_size, preferred_display_size, cluster_indices, num_results=num_results_per_cluster)
        
        # Append cluster recommendations to the overall recommended laptops list
        recommended_laptops.extend(cluster_recommendations)
    
    # Sort recommended laptops based on AI rating (descending order)
    recommended_laptops.sort(key=lambda x: x[1], reverse=True)
    
    return recommended_laptops[:num_results_per_cluster * clusters.n_clusters]

# Function to ask the user for their price limit
def prompt_price_limit():
    while True:
        try:
            price_limit = float(input("\nEnter your price limit: "))
            return price_limit
            break
        except ValueError:
            print("\nPlease enter a valid price.\n")

# Function to prompt the user for their preferred RAM size
def prompt_preferred_ram_size():
    print("\nDo you have a preferred minimum size for RAM?")
    while True:
        response = input("\nEnter 'yes' or 'no': ").lower()
        if response == 'yes':
            # Define the list of available RAM sizes
            available_ram_sizes = [4, 8, 16, 32, 64] # Modify for all sizes
            
            # Prompt the user to select their preferred RAM size from the available options
            print("\nAvailable RAM sizes (in GB):", available_ram_sizes)
            while True:
                try:
                    preferred_ram_size = float(input("\nEnter your preferred minimum RAM size in GB from the list above: "))
                    if preferred_ram_size not in available_ram_sizes:
                        raise ValueError("\nPlease enter a RAM size from the available options.\n")
                    return preferred_ram_size
                except ValueError as e:
                    print(e)
            break
        elif response == 'no':
            return None
        else:
            print("\nPlease enter 'yes' or 'no'.\n")

# Function to prompt the user for their preferred primary storage size
def prompt_preferred_storage_size():
    print("\nDo you have a preferred minimum size for the primary storage?")
    while True:
        response = input("\nEnter 'yes' or 'no': ").lower()
        if response == 'yes':
            # Define the list of available primary storage sizes
            available_storage_sizes = [128, 256, 512, 1024, 2048] # Modify for all sizes
            
            # Prompt the user to select their preferred primary storage size from the available options
            print("\nAvailable primary storage sizes (in GB):", available_storage_sizes)
            while True:
                try:
                    preferred_storage_size = float(input("\nEnter your preferred minimum primary storage size in GB from the list above: "))
                    if preferred_storage_size not in available_storage_sizes:
                        raise ValueError("\nPlease enter a storage size from the available options.\n")
                    return preferred_storage_size
                except ValueError as e:
                    print(e)
            break
        elif response == 'no':
            return None
        else:
            print("\nPlease enter 'yes' or 'no'.\n")

# Function to prompt the user for their preferred display size
def prompt_preferred_display_size():
    print("\nDo you have a preferred display size?")
    while True:
        response = input("\nEnter 'yes' or 'no': ").lower()
        if response == 'yes':
            # Define the list of available display sizes
            available_display_sizes = [13.3, 14, 15.6, 16] 
            
            # Prompt the user to select their preferred display size from the available options
            print("\nAvailable display sizes (in inches):", available_display_sizes)
            while True:
                try:
                    preferred_display_size = float(input("\nEnter your preferred display size in inches from the list above: "))
                    if preferred_display_size not in available_display_sizes:
                        raise ValueError("\nPlease enter a display size from the available options.\n")
                    return preferred_display_size
                except ValueError as e:
                    print(e)
            break
        elif response == 'no':
            return None
        else:
            print("\nPlease enter 'yes' or 'no'.\n")

# Prompt the user for their price limit, preferred RAM size, preferred primary storage size, and preferred display size
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
cluster_indices = np.where(kmeans.labels_ == users_cluster)[0]

# Recommend multiple laptops from each cluster and retrieve their models, the AI rating, and the prices
recommended_laptops = recommend_laptops_from_clusters(max_price_limit, preferred_ram_size, preferred_storage_size, preferred_display_size, kmeans, num_results_per_cluster=1)

if recommended_laptops:
    print("\n\n\nRecommended laptops within your price limit and other preferences\n")
    for i, laptop in enumerate(recommended_laptops, 1):
        print(f"Laptop {i}:")
        print("Model:", laptop[0])
        print("AI Rating:", laptop[1])
        print("Price:", laptop[2])
        print()
else:
    print("Sorry, there are no laptops within your specified criteria.")
