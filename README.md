# Movie Recommender System

## 1. About

I built a content-based Movie Recommendation System that helps users
discover movies similar to one they already like. It uses movie
metadata---overview, genres, keywords, cast, and director---to represent
each movie as text, converts that text into TF-IDF vectors, and ranks
movies using cosine similarity. I built the interface with Streamlit and
use the OMDb API to fetch posters for the recommendations.

**Tech stack:** Python, Pandas, NumPy, scikit-learn, Streamlit,
Requests, OMDb API, Pickle/Joblib, and gdown.

## 2. ML Pipeline

1.  **Load and merge data:** Read movie metadata and credits CSV files,
    then merge them using the movie title.
2.  **Preprocess features:** Extract names from serialized genres,
    keywords, cast, and crew fields; keep the first three cast members
    and extract directors.
3.  **Feature engineering:** Combine overview, genres, keywords, cast,
    and director into one `tags` text field per movie.
4.  **TF-IDF:** Convert each movie's tags into a numerical vector.
    Common terms receive less weight, while more informative terms can
    receive more weight.
5.  **Cosine similarity:** Compare every movie vector with every other
    vector, producing a movie-by-movie similarity matrix.
6.  **Recommend:** Find the selected movie's row, sort candidate movies
    by similarity score in descending order, skip the first result
    (normally the selected movie itself), and return the next five.

**Key idea:** This is content-based filtering. It recommends similar
content; it does not predict a user's rating or use collaborative
filtering.

## 3. Code Walkthrough

### `creating.ipynb` --- Build the recommendation data

- Loads `movies.csv` and `credits.csv` with Pandas and merges them on
  `title`.
- Uses `ast.literal_eval()` to parse serialized list/dictionary fields
  and extract relevant names.
- Cleans missing rows, keeps selected metadata, takes the first three
  cast members, and extracts crew members whose job is `Director`.
- Combines the overview, genres, keywords, cast, and director into
  `tags`.
- Uses `TfidfVectorizer` to create feature vectors and
  `cosine_similarity` to calculate pairwise scores.
- Saves the processed movie dataframe and similarity data as
  serialized artifacts.

### `app.py` --- Run the interactive application

- Loads the processed movie dataframe and saved similarity artifact.
- Uses Streamlit's `selectbox` so the user can choose a movie.
- `recommend(movie)` finds the movie's index, ranks its similarity
  scores, skips the self-match, and collects the top five movie titles
  and poster URLs.
- `fetch_poster(title)` calls the OMDb API and returns a poster URL
  when available.
- Streamlit displays the five recommended titles and posters in
  columns.

**Important:** The dataframe and similarity matrix must preserve the
same movie ordering. Otherwise, similarity scores can be associated with
the wrong titles.

## 4. Limitations and Improvements

- **Error handling:** Add request timeouts, HTTP-status checks,
  API/download error handling, and graceful handling for unknown or
  duplicate titles.
- **Caching:** Cache poster lookups and consider caching loaded
  artifacts to avoid unnecessary repeated work.
- **Scalability:** A dense pairwise similarity matrix requires
  approximately (O(N\^2)) storage. For larger catalogues, consider
  sparse methods or nearest-neighbour retrieval.
- **Evaluation:** Add offline evaluation such as Precision@K,
  Recall@K, or NDCG@K, and assess recommendation relevance and
  diversity.
