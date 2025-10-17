# data_preprocessing.py
# Data Preprocessing Script for MovieLens 100K Recommender System

# ============================================================================
# SECTION 1: IMPORTING LIBRARIES
# ============================================================================
# Import necessary libraries for data processing
import pandas as pd
import numpy as np
# Additional imports will be added as needed
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import os


# ============================================================================
# SECTION 2: LOADING MOVIELENS 100K DATA
# ============================================================================
# Load data from the data/ folder
# Expected files:
#   - data/u.data (ratings)
#   - data/u.user (user information)
#   - data/u.item (movie information)
#   - data/u.genre (genre list)

# TODO: Load ratings data
# ratings = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# TODO: Load user data
# users = pd.read_csv('data/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

# TODO: Load movie/item data
# movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1', names=[...])


# ============================================================================
# SECTION 3: HANDLING MISSING VALUES
# ============================================================================
# Check for and handle missing values in the datasets

# TODO: Identify missing values
# print(ratings.isnull().sum())
# print(users.isnull().sum())
# print(movies.isnull().sum())

# TODO: Implement missing value strategies
# - Drop rows with missing critical values
# - Impute missing values where appropriate
# - Document decisions for missing value handling


# ============================================================================
# SECTION 4: ENCODING CATEGORICAL VARIABLES
# ============================================================================
# Encode categorical features for machine learning models

# TODO: Identify categorical columns
# Categorical features may include:
# - User: gender, occupation, zip_code
# - Movie: genres

# TODO: Apply encoding techniques
# - Label encoding for ordinal variables
# - One-hot encoding for nominal variables
# Example: users['gender_encoded'] = LabelEncoder().fit_transform(users['gender'])


# ============================================================================
# SECTION 5: NORMALIZING FEATURES
# ============================================================================
# Normalize/scale numerical features for better model performance

# TODO: Identify numerical columns requiring normalization
# Numerical features may include:
# - User: age
# - Ratings: rating, timestamp

# TODO: Apply normalization/scaling
# - StandardScaler for features with normal distribution
# - MinMaxScaler for bounded features
# Example: scaler = StandardScaler()
# Example: users['age_normalized'] = scaler.fit_transform(users[['age']])


# ============================================================================
# SECTION 6: SAVING PROCESSED DATA
# ============================================================================
# Save processed datasets to data/processed/ folder

# TODO: Create output directory if it doesn't exist
# import os
# os.makedirs('data/processed/', exist_ok=True)

# TODO: Save processed dataframes
# ratings.to_csv('data/processed/ratings_processed.csv', index=False)
# users.to_csv('data/processed/users_processed.csv', index=False)
# movies.to_csv('data/processed/movies_processed.csv', index=False)

# TODO: Save any preprocessing objects (encoders, scalers) for later use
# import pickle
# with open('data/processed/scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

print("Data preprocessing script structure created. Ready for implementation.")
