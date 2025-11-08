import pandas as pd
import pickle
import streamlit as st
import numpy as np
from scipy.sparse import hstack, csr_matrix
 
st.set_page_config(
    page_title="Amazon Review Prediction",
    layout='centered'
)
 
# Load the model dictionary
try:
    with open("Model/best_model_20251107_042919.pkl","rb") as file:
        model_dict = pickle.load(file)
    
    # Extract components from the dictionary
    model = model_dict.get('model', None)
    vectorizer = model_dict.get('vectorizer', None)
    scaler = model_dict.get('scaler', None)
    feature_names = model_dict.get('feature_names', [])
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    model = None
    vectorizer = None
    scaler = None
    feature_names = []
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None
    vectorizer = None
    scaler = None
    feature_names = []
 
st.title("Amazon Review Prediction")

if model is not None:
    reviews = st.text_input("Enter the Review", key='reviews')
    rating = st.selectbox("Select Rating", [1, 2, 3, 4, 5], key='rating')
   
    if st.button("Predict Review", type="primary"):
        try:
            # Create temporary dataframe for feature extraction
            temp_df = pd.DataFrame({
                'reviews.text': [reviews],
                'reviews.rating': [rating]
            })
           
            # Extract features (simplified version of the notebook's feature extraction)
            temp_df['review_length'] = temp_df['reviews.text'].str.len()
            temp_df['word_count'] = temp_df['reviews.text'].str.split().str.len()
            temp_df['exclamation_count'] = temp_df['reviews.text'].str.count('!')
            temp_df['capital_ratio'] = temp_df['reviews.text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
            temp_df['is_extreme_rating'] = ((temp_df['reviews.rating'] == 1) | (temp_df['reviews.rating'] == 5)).astype(int)
           
            # Sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'best', 'awesome', 'recommend']
            negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'horrible', 'disappointing', 'poor', 'waste']
           
            temp_df['positive_word_count'] = temp_df['reviews.text'].str.lower().apply(lambda x: sum(1 for word in positive_words if word in str(x).split()))
            temp_df['negative_word_count'] = temp_df['reviews.text'].str.lower().apply(lambda x: sum(1 for word in negative_words if word in str(x).split()))
           
            # Rating-text mismatch detection
            temp_df['rating_text_mismatch'] = (
                ((temp_df['reviews.rating'] >= 4) & (temp_df['negative_word_count'] > temp_df['positive_word_count'])) |
                ((temp_df['reviews.rating'] <= 2) & (temp_df['positive_word_count'] > temp_df['negative_word_count']))
            ).astype(int)
           
            # Add default values for missing features
            temp_df['user_review_count'] = 1
            temp_df['user_avg_rating'] = temp_df['reviews.rating']
            temp_df['is_verified_purchase'] = 1
            temp_df['helpful_votes'] = 0
            
            # Ensure reviews.rating is available as a feature if needed
            # (some models might use the raw rating as a feature)
           
            # Transform text using the fitted vectorizer
            x_text = vectorizer.transform([reviews])
           
            # Get numerical features - ensure all features from feature_names are present in the correct order
            if feature_names and len(feature_names) > 0:
                # Create a list to store features in the exact order expected by the scaler
                x_numerical_list = []
                
                for feature_name in feature_names:
                    if feature_name in temp_df.columns:
                        x_numerical_list.append(temp_df[feature_name].fillna(0).values[0])
                    else:
                        # If feature is missing, use default value of 0
                        x_numerical_list.append(0)
                
                # Convert to numpy array and reshape for scaler
                x_numerical = np.array(x_numerical_list).reshape(1, -1)
                x_numerical_scaled = scaler.transform(x_numerical)
            else:
                # Fallback if feature_names is empty
                x_numerical = np.column_stack([
                    temp_df['review_length'].values[0],
                    temp_df['word_count'].values[0],
                    temp_df['exclamation_count'].values[0],
                    temp_df['capital_ratio'].values[0],
                    temp_df['is_extreme_rating'].values[0],
                    temp_df['positive_word_count'].values[0],
                    temp_df['negative_word_count'].values[0],
                    temp_df['rating_text_mismatch'].values[0],
                    temp_df['user_review_count'].values[0],
                    temp_df['user_avg_rating'].values[0],
                    temp_df['is_verified_purchase'].values[0],
                    temp_df['helpful_votes'].values[0]
                ])
                x_numerical_scaled = scaler.transform(x_numerical)
           
            # Combine features
            x_combined = hstack([x_text, csr_matrix(x_numerical_scaled)])
           
            # Make prediction
            prediction = model.predict(x_combined)[0]
            probability = model.predict_proba(x_combined)[0]
           
            # Calculate probabilities
            confidence = max(probability) * 100
            fake_probability = probability[1] * 100 if len(probability) > 1 else probability[0] * 100
            
            # Display results
            st.markdown("### Prediction Result")
           
            if prediction == 1:
                st.error("FAKE REVIEW DETECTED")
            else:
                st.success("GENUINE REVIEW")
           
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.write(f"**Fake Probability:** {fake_probability:.1f}%")

            # Sentiment based on rating
            if rating == 3:
                st.info("Neutral Review")
            elif rating == 1 or rating == 2:
                st.warning("Negative Review")
            elif rating == 4 or rating == 5:
                st.success("Positive Review")
 
 
 
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check your input and try again.")

else:
    st.error("Model not loaded properly")
 
 