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
data = pd.concat([data, keywords_encoded], axis=1)

# Fill any residual NaN values with 0
data = data.fillna(0)

# Create a consolidated genre column from binary genre columns
genre_columns = [col for col in data.columns if col.startswith('genre_')]
data['genres'] = data.apply(lambda row: ', '.join([col.split('_')[1] for col in genre_columns if row[col] == 1]),
                            axis=1)

# Select only numeric data for feature columns
numeric_data = data[numerical_cols]


def get_knn_recommendations(user_profile, numeric_data, data, top_n, min_rating):
    knn = NearestNeighbors(n_neighbors=top_n)
    knn.fit(numeric_data)
    distances, indices = knn.kneighbors([user_profile])

    recommendations = data.iloc[indices[0]]
    recommendations = recommendations[recommendations['rating'] >= min_rating]
    recommendations = recommendations.head(top_n)

    return recommendations[['movie', 'year', 'runtime', 'rating', 'genres']], distances[0]


def plot_radar_chart(user_profile, recommendations, numeric_data):
    # Select only the top 5 recommendations
    recommendations = recommendations.head(5)
    recommendations = recommendations.merge(numeric_data, left_index=True, right_index=True)

    # Get numeric data for recommendations
    rec_numeric_data = recommendations[numerical_cols]

    # Create a radar chart
    categories = ['ope', 'con', 'ext', 'agr', 'neu']
    num_vars = len(categories)

    # Compute angle for each category
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Extend user_profile to match the angles
    user_profile = np.append(user_profile[:5], user_profile[0])

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot user profile
    ax.fill(angles, user_profile, color='blue', alpha=0.25)
    ax.plot(angles, user_profile, color='blue', linewidth=2, linestyle='solid', label='User Profile')

    # Plot recommendations
    for idx, row in recommendations.iterrows():
        movie_profile = row[categories].values.flatten().tolist()
        movie_profile += movie_profile[:1]
        ax.plot(angles, movie_profile, linewidth=1, linestyle='solid', label=f"{row['movie']}",
                color=np.random.rand(3, ))

    # Add labels and title
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)

    plt.title('Top 5 Movie Recommendations', size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()


def plot_scatter_plot(user_profile, recommendations, numeric_data, distances):
    # Merge recommendations with numeric_data to ensure alignment
    recommendations = recommendations.merge(numeric_data, left_index=True, right_index=True).reset_index(drop=True)

    # Get numeric data for recommendations
    rec_numeric_data = recommendations[numerical_cols]

    # Apply PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    pca.fit(numeric_data)
    user_profile_2d = pca.transform([user_profile])
    rec_numeric_data_2d = pca.transform(rec_numeric_data)

    # Create a scatter plot
    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid")

    # Plot user profile
    sns.scatterplot(x=[user_profile_2d[0, 0]], y=[user_profile_2d[0, 1]], s=235, color='blue', marker='o', label='User', legend=None)

    # Plot recommended movies with different colors
    palette = sns.color_palette("husl", len(recommendations))
    sns.scatterplot(x=rec_numeric_data_2d[:, 0], y=rec_numeric_data_2d[:, 1], s=165, color='gold', marker='*',
                    legend=None)

    # Plot lines and similarity scores
    for idx, (x, y) in enumerate(rec_numeric_data_2d):
        plt.plot([user_profile_2d[0, 0], x], [user_profile_2d[0, 1], y], 'k-', lw=0.5)

        # Display distance
        mid_x = (user_profile_2d[0, 0] + x) / 2
        mid_y = (user_profile_2d[0, 1] + y) / 2
        plt.text(mid_x, mid_y, f"{distances[idx]:.2f}", fontsize=8)

    # Annotate movie titles with different colors
    for i, row in recommendations.iterrows():
        plt.text(rec_numeric_data_2d[i, 0], rec_numeric_data_2d[i, 1], f"{row['movie']} ({row['year']})",
                 horizontalalignment='left', size='medium', color='black', weight='semibold')

    plt.title('Top Movie Recommendations')
    plt.show()


if __name__ == '__main__':
    user_id, user_profile, user_data = calculate_scores()

    user_profile = user_profile[1:6]  # Ensure the user profile contains the 5 relevant features (skip age and gender)
    # Save to BigQuery with appending new data rather than replacing
    to_gbq(
        user_data,
        destination_table=f"{project_id}.{dataset_id}.movies_user_profiles_features",
        project_id=project_id,
        if_exists="append"
    )
    recommendations_knn, distances = get_knn_recommendations(user_profile, numeric_data, data, 10, 5.0)
    print("\n\nTop 10 Movie Recommendations (Rating >= 5):")
    print(recommendations_knn.to_string(index=False, justify='left'))
    plot_radar_chart(user_profile, recommendations_knn, numeric_data)
    plot_scatter_plot(user_profile, recommendations_knn, numeric_data, distances)
