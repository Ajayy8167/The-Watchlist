🎬 The Watchlist: A Movie Recommendation System

This project is a high-performance, content-based movie recommendation system built with Python and Streamlit. It provides instant, reliable movie suggestions based on a rich, pre-processed dataset. The application features a stylish, user-friendly interface with advanced filtering options, guaranteeing a seamless and fast user experience.

✨ Features

- Content-Based Recommendations: Suggests movies based on similarity of genre, plot, director, and cast.
- 100% Offline & Instant: All data is pre-processed, eliminating the need for live API calls and ensuring zero latency.
- Interactive UI: A clean and modern interface built with Streamlit, featuring a 5x2 grid for recommendations.
- Advanced Filtering: Allows users to sort and refine the movie list by genre, release year, and rating.
- Rich Details: Each recommendation includes a poster, plot summary, rating, and other key details.

🛠️ Tech Stack

Language: Python

Libraries:
- Streamlit: For the interactive web application interface.
- Pandas: For data manipulation and pre-processing.
- Scikit-learn: For building the recommendation engine (TF-IDF, Cosine Similarity).
- Requests: (Used in the pre-computation script) for making API calls.

🚀 Setup and Installation

To run this project locally, please follow these steps:

Clone the Repository:
```bash
git clone https://your-repository-link.git
cd your-project-folder

Install Dependencies:
Make sure you have a requirements.txt file with the following content:

streamlit
pandas
scikit-learn
requests
Then, install the required libraries:


pip install -r requirements.txt
Set Up the Data:
Create a folder named data in the main project directory.

Place your movie dataset (new_movies_full.csv) inside the data folder.

Run the Pre-computation Script (One-Time Only):

python precompute.py

Run the Application:
streamlit run app.py


🎯 Usage
Open the application in your browser.
Use the filters in the sidebar to narrow down the movie list.
Select a movie you like from the dropdown.
Click "Get Recommendations" to see 10 similar movies.