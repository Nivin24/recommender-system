# data_preprocessing.py
# Data Preprocessing Script for MovieLens 100K Recommender System
# ============================================================================
# SECTION 1: IMPORTING LIBRARIES
# ============================================================================
# Import necessary libraries for data processing
import pandas as pd
import numpy as np
import os
# Additional imports will be added as needed
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# ============================================================================
# SECTION 2: LOADING MOVIELENS 100K DATA
# ============================================================================
# Load data from the data/ folder
# Expected files:
#   - data/u.data (ratings)
#   - data/u.user (user information)
#   - data/u.item (movie information)
#   - data/u.genre (genre list)
print("Loading MovieLens 100K datasets...\n")
# Load ratings data
ratings = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
print("Ratings Data:")
print(ratings.head())
print(f"Shape: {ratings.shape}\n")
# Load user data
users = pd.read_csv('data/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
print("Users Data:")
print(users.head())
print(f"Shape: {users.shape}\n")
# Load movie/item data
movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1', names=movie_columns)
print("Movies Data:")
print(movies.head())
print(f"Shape: {movies.shape}\n")
# ============================================================================
# SECTION 3: SAVING INITIAL LOADED DATA
# ============================================================================
# Save initial loaded versions to data/processed/ folder
print("Saving initial loaded data to data/processed/...\n")
# Create output directory if it doesn't exist
os.makedirs('data/processed/', exist_ok=True)
# Save initial loaded dataframes
ratings.to_csv('data/processed/ratings_initial.csv', index=False)
users.to_csv('data/processed/users_initial.csv', index=False)
movies.to_csv('data/processed/movies_initial.csv', index=False)
print("Initial data saved successfully!")
print(f"Files saved to data/processed/:")
print(f"  - ratings_initial.csv")
print(f"  - users_initial.csv")
print(f"  - movies_initial.csv")
# ============================================================================
# SECTION 4: HANDLING MISSING VALUES
# ============================================================================
# Check for and handle missing values in the datasets
print("\n" + "="*80)
print("CHECKING FOR MISSING VALUES")
print("="*80 + "\n")
# Check missing values in ratings dataframe
print("Missing Values in Ratings Data:")
print(ratings.isnull().sum())
print(f"Total missing values: {ratings.isnull().sum().sum()}\n")
# Check missing values in users dataframe
print("Missing Values in Users Data:")
print(users.isnull().sum())
print(f"Total missing values: {users.isnull().sum().sum()}\n")
# Check missing values in movies dataframe
print("Missing Values in Movies Data:")
print(movies.isnull().sum())
print(f"Total missing values: {movies.isnull().sum().sum()}\n")
# Strategy for handling missing values:
# - Ratings data: Drop any rows with missing user_id, movie_id, or rating values
# - Users data: Drop rows with missing user_id, impute gender/occupation with mode, age with median
# - Movies data: Drop missing titles, impute missing genre columns with 0, release_date with placeholder
print("\n" + "="*80)
print("HANDLING MISSING VALUES")
print("="*80 + "\n")
# Handle missing values in ratings dataframe
print("Processing Ratings Data...")
ratings_before = len(ratings)
ratings = ratings.dropna(subset=['user_id', 'movie_id', 'rating'])
ratings_after = len(ratings)
print(f"  Dropped {ratings_before - ratings_after} rows with missing critical values")
print(f"  Ratings shape after cleaning: {ratings.shape}\n")
# Handle missing values in users dataframe
print("Processing Users Data...")
users_before = len(users)
# Drop rows with missing user_id
users = users.dropna(subset=['user_id'])
print(f"  Dropped {users_before - len(users)} rows with missing user_id")
# Impute missing gender with mode
if users['gender'].isnull().sum() > 0:
    gender_mode = users['gender'].mode()[0]
    users['gender'] = users['gender'].fillna(gender_mode)
    print(f"  Imputed missing gender values with mode: {gender_mode}")
# Impute missing occupation with mode
if users['occupation'].isnull().sum() > 0:
    occupation_mode = users['occupation'].mode()[0]
    users['occupation'] = users['occupation'].fillna(occupation_mode)
    print(f"  Imputed missing occupation values with mode: {occupation_mode}")
# Impute missing age with median
if users['age'].isnull().sum() > 0:
    age_median = users['age'].median()
    users['age'] = users['age'].fillna(age_median)
    print(f"  Imputed missing age values with median: {age_median}")
print(f"  Users shape after cleaning: {users.shape}\n")
# Handle missing values in movies dataframe
print("Processing Movies Data...")
movies_before = len(movies)
# Drop rows with missing titles
movies = movies.dropna(subset=['title'])
print(f"  Dropped {movies_before - len(movies)} rows with missing titles")
# Impute missing genre columns (binary columns) with 0
genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
for col in genre_columns:
    if movies[col].isnull().sum() > 0:
        movies[col] = movies[col].fillna(0)
        print(f"  Imputed missing values in {col} with 0")
# Impute missing release_date with placeholder
if movies['release_date'].isnull().sum() > 0:
    movies['release_date'] = movies['release_date'].fillna('01-Jan-1970')
    print(f"  Imputed {movies['release_date'].isna().sum()} missing release_date values with '01-Jan-1970'")
print(f"  Movies shape after cleaning: {movies.shape}\n")
# ============================================================================
# SECTION 5: ENCODING CATEGORICAL VARIABLES
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
# SECTION 6: NORMALIZING FEATURES
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
# SECTION 7: SAVING PROCESSED DATA
# ============================================================================
# Save processed datasets to data/processed/ folder
print("\n" + "="*80)
print("SAVING PROCESSED DATA")
print("="*80 + "\n")
ratings.to_csv('data/processed/ratings_processed.csv', index=False)
users.to_csv('data/processed/users_processed.csv', index=False)
movies.to_csv('data/processed/movies_processed.csv', index=False)
print("Processed data saved successfully!")
print(f"Files saved to data/processed/:")
print(f"  - ratings_processed.csv ({len(ratings)} rows)")
print(f"  - users_processed.csv ({len(users)} rows)")
print(f"  - movies_processed.csv ({len(movies)} rows)")
print("\nData preprocessing completed successfully!")
# TODO: Save any preprocessing objects (encoders, scalers) for later use
# import pickle
# with open('data/processed/scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)
