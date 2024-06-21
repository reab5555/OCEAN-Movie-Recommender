import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
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
numerical_cols = ['gender', 'ope', 'con', 'ext', 'agr', 'neu']
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


def get_recommendations(user_profile, feature_data, data, top_n, min_rating):
    # Extend user profile with zeros for keyword features
    extended_user_profile = np.concatenate((user_profile, np.zeros(len(feature_data.columns) - len(user_profile))))

    # Calculate similarity scores
    similarity_scores = cosine_similarity([extended_user_profile], feature_data)

    # Add similarity scores to the data
    data['Recommendation_Score'] = similarity_scores[0] * 100

    # Filter by minimum rating and sort by recommendation score
    filtered_data = data[data['rating'] >= min_rating]
    recommendations = filtered_data.sort_values(by='Recommendation_Score', ascending=False).head(top_n)

    # Format recommendation score
    recommendations['Recommendation_Score'] = recommendations['Recommendation_Score'].apply(lambda x: f"{x:.0f}%")

    return recommendations[['Recommendation_Score', 'movie', 'year', 'runtime', 'rating', 'genres']]


def plot_radar_chart(user_profile, recommendations, feature_data):
    # Select only the top 5 recommendations
    recommendations = recommendations.head(5)
    recommendations = recommendations.merge(feature_data, left_index=True, right_index=True)

    # Get numeric data for recommendations
    rec_numeric_data = recommendations[numerical_cols]

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
    for idx, row in recommendations.iterrows():
        movie_profile = row[categories].values.flatten().tolist()
        movie_profile += movie_profile[:1]
        similarity_score = cosine_similarity([user_profile[:-1]], [row[categories]])[0][0] * 100
        ax.plot(angles, movie_profile, linewidth=1, linestyle='solid',
                label=f"{row['movie']} ({similarity_score:.1f}%)", color=np.random.rand(3, ))

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

    recommendations_similarity = get_recommendations(user_profile, feature_data, data, 10, 5.0)
    print("\n\nTop 10 Movie Recommendations (Rating >= 5):")
    print(recommendations_similarity.to_string(index=False, justify='left'))
    plot_radar_chart(user_profile, recommendations_similarity, feature_data)
