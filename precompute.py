import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import ast # To safely evaluate string-based lists/dicts

print("Starting pre-computation process for the new dataset...")

# --- CONFIGURATION ---
DATA_FOLDER = "data"
INPUT_CSV = "new_movies_full.csv"
OUTPUT_PKL = "cosine_sim.pkl"

input_path = os.path.join(DATA_FOLDER, INPUT_CSV)
output_path = os.path.join(DATA_FOLDER, OUTPUT_PKL)

# 1. Load the new dataset
df = pd.read_csv(input_path)
df.drop_duplicates(subset='title', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Loaded {len(df)} unique movies from '{INPUT_CSV}'.")

# 2. Helper functions to extract valuable text from columns
def get_names(text):
    """Extracts names from stringified lists of dictionaries."""
    try:
        # Safely evaluate the string to a list
        items = ast.literal_eval(text)
        if isinstance(items, list):
            return ' '.join([i['name'] for i in items])
    except (ValueError, SyntaxError):
        pass
    return ''

def get_director(text):
    """Extracts the director's name."""
    try:
        items = ast.literal_eval(text)
        if isinstance(items, list):
            for i in items:
                if i.get('job') == 'Director':
                    return i['name']
    except (ValueError, SyntaxError):
        pass
    return ''

# 3. Create the "soup" of features for recommendation
print("Preparing features for the recommendation engine...")
df['genres'] = df['genres'].apply(get_names)
df['keywords'] = df['keywords'].apply(get_names)
df['director'] = df['credits'].apply(get_director)

# For credits, let's take the top 3 actors
def get_top_cast(text):
    try:
        items = ast.literal_eval(text)
        if isinstance(items, list):
            return ' '.join([i['name'] for i in items[:3]])
    except (ValueError, SyntaxError):
        pass
    return ''
df['cast'] = df['credits'].apply(get_top_cast)

# Fill any missing text data
text_features = ['overview', 'genres', 'keywords', 'director', 'cast']
for feature in text_features:
    df[feature] = df[feature].fillna('')

# Combine all features into a single "soup"
df['soup'] = df['title'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']
print("✅ Feature soup created.")

# 4. Calculate and save the similarity matrix
print("Calculating similarity matrix...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix).astype(np.float16)
print("✅ Similarity matrix calculated.")

# 5. Save the matrix
with open(output_path, 'wb') as f:
    pickle.dump(cosine_sim, f)

print(f"✅ Similarity matrix saved to '{output_path}'.")
print("\n🎉 Pre-computation finished! You can now run the final app.py. 🎉")