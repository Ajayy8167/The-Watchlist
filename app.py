import streamlit as st
import pandas as pd
import pickle
import base64
import os
import ast

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="The Watchlist")

# --- ASSETS (100% LOCAL) ---
@st.cache_data
def get_local_image_as_base64(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

logo_path = "logo_the_watchlist.png"
placeholder_path = "movieimage.jpg"
logo_base64 = get_local_image_as_base64(logo_path)
placeholder_base64 = get_local_image_as_base64(placeholder_path)


# --- DATA LOADING (100% LOCAL AND OFFLINE) ---
@st.cache_data
def load_data():
    path = os.path.join("data", "new_movies_full.csv")
    if not os.path.exists(path):
        st.error(f"CRITICAL ERROR: '{os.path.basename(path)}' not found in the 'data' folder.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    base_poster_url = "https://image.tmdb.org/t/p/w500"
    df['full_poster_path'] = df['poster_path'].apply(lambda x: base_poster_url + str(x) if pd.notna(x) else '')
    df['description'] = df['overview'].fillna('Plot not available.')
    df['tagline'] = df['tagline'].fillna('')

    def get_genre_names(text):
        try:
            genres = ast.literal_eval(text)
            if isinstance(genres, list):
                return [g['name'] for g in genres]
        except (ValueError, SyntaxError):
            pass
        return []
    df['genres_list'] = df['genres'].apply(get_genre_names)

    if placeholder_base64:
        df['full_poster_path'] = df['full_poster_path'].apply(lambda x: x if x.startswith('http') else f"data:image/jpeg;base64,{placeholder_base64}")
    else:
        df['full_poster_path'] = df['full_poster_path'].apply(lambda x: x if x.startswith('http') else "https://i.imgur.com/sA5zP58.png")

    return df

@st.cache_resource
def load_similarity_matrix():
    path = os.path.join("data", "cosine_sim.pkl")
    if not os.path.exists(path):
        st.error("CRITICAL ERROR: 'cosine_sim.pkl' not found. Please run the `precompute.py` script once to create it.")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

df = load_data()
cosine_sim = load_similarity_matrix()

if not df.empty and cosine_sim is not None:
    df.dropna(subset=['title'], inplace=True)
    df.drop_duplicates(subset='title', inplace=True)
    df.reset_index(drop=True, inplace=True)
    indices = pd.Series(df.index, index=df['title'])
else:
    st.stop()


# --- RECOMMENDATION LOGIC (INSTANT) ---
def get_recommendations(title):
    if title not in indices: return pd.DataFrame()
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores if i[0] < len(df)]
    return df.iloc[movie_indices] if movie_indices else pd.DataFrame()


# --- UI & STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E0E0E; color: #FFFFFF; }
    h1, h2, h3 { color: #E50914; text-transform: uppercase; }
    .movie-card {
        background-color: #1A1A1A; border-radius: 10px; border: 1px solid #2C2C2C;
        padding: 1rem; margin-bottom: 1rem; transition: transform 0.2s, box-shadow 0.2s;
    }
    .movie-card:hover { transform: scale(1.05); box-shadow: 0px 10px 20px rgba(0,0,0,0.5); }
    .poster-box { height: 300px; width: 100%; overflow: hidden; border-radius: 8px; }
    .poster-img { width: 100%; height: 100%; object-fit: cover; }
    .info-box { padding-top: 1rem; text-align: center; }
    .movie-title { font-size: 1.1em; font-weight: bold; height: 45px; overflow: hidden; margin-bottom: 0.5rem; }
    .movie-rating { font-size: 1em; font-weight: bold; color: #FFD700; margin-bottom: 0.5rem; }
    .movie-details { font-size: 0.9em; color: #AAAAAA; }
    .stRadio > div { display: flex; justify-content: center; }
    .st-emotion-cache-163ttbj {
        border: 1px solid #2C2C2C; background-color: #1A1A1A;
        border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
if logo_base64:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem;">
            <div style="flex: 1;">
                <img src="data:image/png;base64,{logo_base64}" width="100">
            </div>
            <div style="flex: 2; text-align: center;">
                <h1 style="margin: 0;">The Watchlist</h1>
            </div>
            <div style="flex: 1;"></div>
        </div>
        """, unsafe_allow_html=True
    )
else:
    st.title("🎬 The Watchlist")


# --- SIDEBAR FILTERS ---
st.sidebar.title('Find Your Perfect Movie')

with st.sidebar.container():
    st.header('📈 Sort By')
    sort_option = st.selectbox('Sort by:', ('Alphabetical (A-Z)', 'Popularity', 'Rating', 'Newest First'), label_visibility="collapsed")

with st.sidebar.container():
    st.header('🎭 Refine By')
    
    all_genres = set(g for genres in df['genres_list'] for g in genres)
    selected_genre = st.selectbox('Genre', ['All'] + sorted(list(all_genres)))

    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df.dropna(subset=['release_year'], inplace=True)
    min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())
    
    st.write("Release Year")
    year_range_tuple = st.slider('Select a range of years', min_year, max_year, (min_year, max_year))
    col1, spacer, col2 = st.columns([4, 1, 4])
    with col1:
        start_year = st.number_input('From', value=year_range_tuple[0], min_value=min_year, max_value=max_year)
    with spacer:
        st.markdown("<p style='text-align: center; margin-top: 2.5em;'>to</p>", unsafe_allow_html=True)
    with col2:
        end_year = st.number_input('To', value=year_range_tuple[1], min_value=min_year, max_value=max_year)
    year_range = (start_year, end_year)
    
    rating_range = st.slider('Movie Ratings ⭐', 0.0, 10.0, 0.0, 0.5, format="%.1f")


# --- FILTERING LOGIC ---
filtered_df = df.copy()
if selected_genre != 'All':
    filtered_df = filtered_df[filtered_df['genres_list'].apply(lambda genres: selected_genre in genres)]

filtered_df = filtered_df[
    (filtered_df['release_year'] >= year_range[0]) &
    (filtered_df['release_year'] <= year_range[1])
]
filtered_df = filtered_df[filtered_df['vote_average'] >= rating_range]
if sort_option == 'Popularity': filtered_df = filtered_df.sort_values(by='popularity', ascending=False)
elif sort_option == 'Rating': filtered_df = filtered_df.sort_values(by='vote_average', ascending=False)
elif sort_option == 'Newest First': filtered_df = filtered_df.sort_values(by='release_date', ascending=False)
else: filtered_df = filtered_df.sort_values(by='title')
movie_list = filtered_df['title'].tolist()


# --- MAIN PAGE ---
if movie_list:
    selected_movie = st.selectbox("Select a movie you like to get recommendations", movie_list, label_visibility="collapsed")
    if st.button('Get Recommendations'):
        if selected_movie:
            with st.spinner('Finding your next favorite movies... 🍿'):
                recommendations = get_recommendations(selected_movie)
            if not recommendations.empty:
                st.header(f'Because you watched "{selected_movie}"...')
                def movie_card(movie_data):
                    first_genre = movie_data.genres_list[0] if movie_data.genres_list else ''
                    st.markdown(f"""
                        <div class="movie-card">
                            <div class="poster-box">
                                <img src="{movie_data.full_poster_path}" class="poster-img">
                            </div>
                            <div class="info-box">
                                <p class="movie-title">{movie_data.title}</p>
                                <p class="movie-rating">{movie_data.vote_average:.1f} ⭐</p>
                                <p class="movie-details">{int(movie_data.release_year)} | {first_genre}</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Show Plot"):
                        st.markdown(f"*{movie_data.tagline}*")
                        st.write(movie_data.description)
                    st.radio("Rate it:", ('👍', '👎'), key=f"rating_{movie_data.id}", horizontal=True, label_visibility="collapsed")

                first_five = recommendations.head(5)
                next_five = recommendations.iloc[5:10]
                cols_row1 = st.columns(5)
                for i, movie in enumerate(first_five.itertuples(index=False)):
                    with cols_row1[i]:
                        movie_card(movie)
                if len(next_five) > 0:
                    cols_row2 = st.columns(5)
                    for i, movie in enumerate(next_five.itertuples(index=False)):
                        with cols_row2[i]:
                            movie_card(movie)
        else:
            st.warning("Please select a movie first.")
else:
    st.warning("No movies match your filter criteria. Please adjust the filters.")