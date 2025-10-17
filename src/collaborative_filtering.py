# ============================================================================
# collaborative_filtering.py
# ============================================================================
# Collaborative filtering module for the recommender system
# Implements SVD-based recommendation algorithms using TruncatedSVD
# ============================================================================

# ----------------------------------------------------------------------------
# Section 1: Import Required Libraries
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------
# Section 2: Load Processed Data
# ----------------------------------------------------------------------------
def load_processed_data():
    """
    Load processed ratings, users, and movies data.
    
    Returns:
        ratings_df (DataFrame): Processed ratings data
        users_df (DataFrame): Processed users data
        movies_df (DataFrame): Processed movies data
    """
    print("Loading processed data...")
    
    # Load processed datasets
    ratings_df = pd.read_csv('data/processed/ratings_processed.csv')
    users_df = pd.read_csv('data/processed/users_processed.csv')
    movies_df = pd.read_csv('data/processed/movies_processed.csv')
    
    print(f"Loaded {len(ratings_df)} ratings, {len(users_df)} users, {len(movies_df)} movies")
    return ratings_df, users_df, movies_df

# ----------------------------------------------------------------------------
# Section 3: Build User-Item Ratings Matrix
# ----------------------------------------------------------------------------
def build_user_item_matrix(ratings_df):
    """
    Build a user-item ratings matrix from ratings data.
    
    Args:
        ratings_df (DataFrame): Ratings data with userId, movieId, and rating columns
    
    Returns:
        user_item_matrix (DataFrame): User-item ratings matrix (users x movies)
        user_mapper (dict): Mapping from userId to matrix row index
        movie_mapper (dict): Mapping from movieId to matrix column index
    """
    print("Building user-item matrix...")
    
    # Create pivot table with users as rows and movies as columns
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')
    
    # Fill NaN values with 0 (no rating)
    user_item_matrix = user_item_matrix.fillna(0)
    
    # Create mappings from IDs to matrix indices
    user_mapper = {user_id: idx for idx, user_id in enumerate(user_item_matrix.index)}
    movie_mapper = {movie_id: idx for idx, movie_id in enumerate(user_item_matrix.columns)}
    
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    return user_item_matrix, user_mapper, movie_mapper

# ----------------------------------------------------------------------------
# Section 4: SVD Implementation using TruncatedSVD
# ----------------------------------------------------------------------------
def train_svd_model(user_item_matrix, n_components=20):
    """
    Train a TruncatedSVD model for collaborative filtering.
    
    Args:
        user_item_matrix (DataFrame): User-item ratings matrix
        n_components (int): Number of latent factors/components
    
    Returns:
        svd_model: Fitted TruncatedSVD model
        user_factors: Transformed user matrix (U * S)
        movie_factors: Components matrix (V.T)
    """
    print(f"Training SVD model with {n_components} components...")
    
    # Initialize TruncatedSVD
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    
    # Fit and transform the user-item matrix
    user_factors = svd_model.fit_transform(user_item_matrix)
    
    # Get movie factors (components)
    movie_factors = svd_model.components_
    
    print(f"SVD training completed. Explained variance ratio: {svd_model.explained_variance_ratio_.sum():.4f}")
    
    return svd_model, user_factors, movie_factors

def predict_rating(user_id, movie_id, svd_model, user_factors, movie_factors, 
                  user_mapper, movie_mapper, user_item_matrix):
    """
    Predict a rating for a given user-movie pair using SVD.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        svd_model: Fitted TruncatedSVD model
        user_factors: Transformed user matrix
        movie_factors: Movie components matrix
        user_mapper: User ID to matrix index mapping
        movie_mapper: Movie ID to matrix index mapping
        user_item_matrix: Original user-item matrix
    
    Returns:
        predicted_rating (float): Predicted rating value
    """
    # Check if user and movie exist in our mappings
    if user_id not in user_mapper or movie_id not in movie_mapper:
        # Return average rating if user or movie not in training data
        return user_item_matrix[user_item_matrix > 0].mean().mean()
    
    # Get matrix indices
    user_idx = user_mapper[user_id]
    movie_idx = movie_mapper[movie_id]
    
    # Reconstruct rating using dot product of user and movie latent factors
    predicted_rating = np.dot(user_factors[user_idx], movie_factors[:, movie_idx])
    
    # Clip rating to valid range (typically 1-5 for movie ratings)
    predicted_rating = np.clip(predicted_rating, 1, 5)
    
    return predicted_rating

# ----------------------------------------------------------------------------
# Section 5: Model Persistence
# ----------------------------------------------------------------------------
def save_model(svd_model, user_factors, movie_factors, user_mapper, movie_mapper, 
               user_item_matrix, model_dir='models/'):
    """
    Save the trained SVD model and all components to disk.
    
    Args:
        svd_model: Fitted TruncatedSVD model
        user_factors: Transformed user matrix
        movie_factors: Movie components matrix
        user_mapper: User ID to index mapping
        movie_mapper: Movie ID to index mapping
        user_item_matrix: Original user-item matrix
        model_dir (str): Directory to save model files
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save all components
    model_data = {
        'svd_model': svd_model,
        'user_factors': user_factors,
        'movie_factors': movie_factors,
        'user_mapper': user_mapper,
        'movie_mapper': movie_mapper,
        'user_item_matrix': user_item_matrix
    }
    
    model_path = os.path.join(model_dir, 'collaborative_filtering_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_path}")

def load_model(model_path='models/collaborative_filtering_model.pkl'):
    """
    Load a trained collaborative filtering model from disk.
    
    Args:
        model_path (str): Path to saved model
    
    Returns:
        dict: Dictionary containing all model components
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model_data
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None

# ----------------------------------------------------------------------------
# Section 6: Recommendation Generation
# ----------------------------------------------------------------------------
def get_top_n_recommendations(user_id, n=10, svd_model=None, user_factors=None, 
                             movie_factors=None, user_mapper=None, movie_mapper=None,
                             user_item_matrix=None, movies_df=None):
    """
    Get top N movie recommendations for a user.
    
    Args:
        user_id: User ID to generate recommendations for
        n (int): Number of recommendations to generate
        svd_model, user_factors, movie_factors, user_mapper, movie_mapper, user_item_matrix: Model components
        movies_df: Movies dataframe for title lookup
    
    Returns:
        recommendations (DataFrame): Top N recommendations with movie details
    """
    if user_id not in user_mapper:
        print(f"User {user_id} not found in training data")
        return pd.DataFrame()
    
    # Get all movies the user hasn't rated
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # Predict ratings for all unrated movies
    predictions = []
    for movie_id in unrated_movies:
        pred_rating = predict_rating(user_id, movie_id, svd_model, user_factors, 
                                   movie_factors, user_mapper, movie_mapper, user_item_matrix)
        predictions.append((movie_id, pred_rating))
    
    # Sort by predicted rating and get top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = predictions[:n]
    
    # Create recommendations dataframe
    rec_df = pd.DataFrame(top_recommendations, columns=['movieId', 'predicted_rating'])
    
    # Add movie titles if movies_df is provided
    if movies_df is not None:
        rec_df = rec_df.merge(movies_df[['movieId', 'title']], on='movieId', how='left')
    
    return rec_df

# ----------------------------------------------------------------------------
# Section 7: Model Evaluation
# ----------------------------------------------------------------------------
def evaluate_model(svd_model, user_factors, movie_factors, user_mapper, movie_mapper,
                  user_item_matrix, test_ratings):
    """
    Evaluate the SVD model on test data.
    
    Args:
        svd_model, user_factors, movie_factors, user_mapper, movie_mapper, user_item_matrix: Model components
        test_ratings (DataFrame): Test ratings data
    
    Returns:
        rmse (float): Root Mean Square Error
    """
    predictions = []
    actuals = []
    
    for _, row in test_ratings.iterrows():
        pred = predict_rating(row['userId'], row['movieId'], svd_model, user_factors,
                            movie_factors, user_mapper, movie_mapper, user_item_matrix)
        predictions.append(pred)
        actuals.append(row['rating'])
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return rmse

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------
def main():
    """
    Main function to train and test the collaborative filtering model.
    """
    print("=" * 80)
    print("Collaborative Filtering with TruncatedSVD")
    print("=" * 80)
    
    try:
        # Load processed data
        ratings_df, users_df, movies_df = load_processed_data()
        
        # Build user-item matrix
        user_item_matrix, user_mapper, movie_mapper = build_user_item_matrix(ratings_df)
        
        # Train SVD model
        svd_model, user_factors, movie_factors = train_svd_model(user_item_matrix, n_components=20)
        
        # Save the model
        save_model(svd_model, user_factors, movie_factors, user_mapper, 
                  movie_mapper, user_item_matrix)
        
        # Test predictions with some example user-movie pairs
        print("\nExample Predictions:")
        print("-" * 50)
        
        # Get a few sample users and movies
        sample_users = list(user_mapper.keys())[:5]
        sample_movies = list(movie_mapper.keys())[:5]
        
        for user_id in sample_users:
            for movie_id in sample_movies:
                pred_rating = predict_rating(user_id, movie_id, svd_model, user_factors,
                                           movie_factors, user_mapper, movie_mapper, 
                                           user_item_matrix)
                
                # Get actual rating if it exists
                actual_rating = user_item_matrix.loc[user_id, movie_id] if movie_id in user_item_matrix.columns else 0
                
                movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0] if len(movies_df[movies_df['movieId'] == movie_id]) > 0 else f"Movie {movie_id}"
                
                print(f"User {user_id}, {movie_title}: Predicted={pred_rating:.2f}, Actual={actual_rating}")
                break  # Only show one prediction per user for brevity
        
        # Generate sample recommendations
        print("\nSample Recommendations for User", sample_users[0], ":")
        print("-" * 50)
        
        recommendations = get_top_n_recommendations(
            sample_users[0], n=5, svd_model=svd_model, user_factors=user_factors,
            movie_factors=movie_factors, user_mapper=user_mapper, 
            movie_mapper=movie_mapper, user_item_matrix=user_item_matrix,
            movies_df=movies_df
        )
        
        if not recommendations.empty:
            for _, rec in recommendations.iterrows():
                movie_title = rec['title'] if 'title' in rec else f"Movie {rec['movieId']}"
                print(f"{movie_title}: {rec['predicted_rating']:.2f}")
        
        print("\nCollaborative filtering model training completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        print("Make sure the processed data files exist in data/processed/")

if __name__ == "__main__":
    main()
