import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from google.cloud import bigquery
from pandas_gbq import to_gbq
from Questionnaire import calculate_scores
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.decomposition import PCA

# Set pandas options for better data viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# Initialize the BigQuery client
client = bigquery.Client()
project_id = "python-code-running"
dataset_id = "moviesets"
table_id = "movies_big_five_set"
# Load data from BigQuery
query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
data = client.query(query).to_dataframe()

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['ope', 'con', 'ext', 'agr', 'neu']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Filter out movies with 'None' keywords
data = data[data['keywords'] != 'None']

# Handling keywords - transform into one-hot encoded columns
keywords_encoded = data['keywords'].str.get_dummies(sep=', ')

# Define genre columns
genre_columns = ['genre_family', 'genre_fantasy', 'genre_sport', 'genre_biography', 'genre_crime',
                 'genre_romance', 'genre_animation', 'genre_music', 'genre_comedy', 'genre_war',
                 'genre_sci_fi', 'genre_horror', 'genre_western', 'genre_thriller', 'genre_mystery',
                 'genre_drama', 'genre_action', 'genre_history', 'genre_documentary', 'genre_musical',
                 'genre_adventure']

# Create a consolidated genre column from binary genre columns
data['genres'] = data[genre_columns].apply(
    lambda row: ', '.join([col.split('_')[1] for col, value in row.items() if value == 1]),
    axis=1
)

# Combine numerical features, keywords, and genres
feature_data = pd.concat([data[numerical_cols], keywords_encoded, data[genre_columns]], axis=1)

# Fill any residual NaN values with 0
feature_data = feature_data.fillna(0)

def get_knn_recommendations(user_profile, feature_data, data, top_n, min_rating):
    # Extend user profile with zeros for keyword and genre features
    extended_user_profile = np.concatenate((user_profile, np.zeros(len(feature_data.columns) - len(user_profile))))

    knn = NearestNeighbors(n_neighbors=top_n)
    knn.fit(feature_data)
    distances, indices = knn.kneighbors([extended_user_profile])

    recommendations = data.iloc[indices[0]]
    recommendations = recommendations[recommendations['rating'] >= min_rating]
    recommendations = recommendations.head(top_n)

    return recommendations[['movie', 'year', 'runtime', 'rating', 'genres']], distances[0]

def plot_radar_chart(user_profile, recommendations, feature_data):
    # Select only the top 5 recommendations and the numerical columns
    recommendations = recommendations.head(5)
    rec_numeric_data = feature_data.loc[recommendations.index, numerical_cols]

    # Create a radar chart
    categories = numerical_cols
    num_vars = len(categories)

    # Compute angle for each category
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Extend user_profile to match the angles
    user_profile = np.append(user_profile, user_profile[0])

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot user profile
    ax.fill(angles, user_profile, color='blue', alpha=0.25)
    ax.plot(angles, user_profile, color='blue', linewidth=2, linestyle='solid', label='User Profile')

    # Plot recommendations
    for idx, row in rec_numeric_data.iterrows():
        movie_profile = row.values.flatten().tolist()
        movie_profile += movie_profile[:1]
        ax.plot(angles, movie_profile, linewidth=1, linestyle='solid', label=f"{recommendations.loc[idx, 'movie']}",
                color=np.random.rand(3, ))

    # Add labels and title
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)

    plt.title('Top 5 Movie Recommendations (Personality Features)', size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()


if __name__ == '__main__':
    user_id, user_profile, user_data = calculate_scores()

    user_profile = user_profile[1:7]  # Ensure the user profile contains the 6 relevant features (skip age)
    # Save to BigQuery with appending new data rather than replacing
    to_gbq(
        user_data,
        destination_table=f"{project_id}.{dataset_id}.movies_user_profiles_features",
        project_id=project_id,
        if_exists="append"
    )
    recommendations_knn, distances = get_knn_recommendations(user_profile, feature_data, data, 10, 5.0)
    print("\n\nTop 10 Movie Recommendations (Rating >= 5):")
    print(recommendations_knn.to_string(index=False, justify='left'))
    plot_radar_chart(user_profile, recommendations_knn, feature_data)
