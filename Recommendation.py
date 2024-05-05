import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery
from pandas_gbq import to_gbq
from Questionnaire import calculate_scores

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
numerical_cols = ['gender', 'age', 'ope', 'con', 'ext', 'agr', 'neu']
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


def get_recommendations(user_profile, numeric_data, data, top_n, min_rating):
    extended_user_profile = np.concatenate((user_profile, np.zeros(len(numeric_data.columns) - len(user_profile))))
    similarity_scores = cosine_similarity([extended_user_profile], numeric_data)
    data['Recommendation_Score'] = similarity_scores[0] * 100
    filtered_data = data[data['rating'] >= min_rating]
    recommendations = filtered_data.sort_values(by='Recommendation_Score', ascending=False).head(top_n)
    recommendations['Recommendation_Score'] = recommendations['Recommendation_Score'].apply(lambda x: f"{x:.0f}%")
    return recommendations[['Recommendation_Score', 'movie', 'year', 'runtime', 'rating', 'genres']]


if __name__ == '__main__':
    user_id, user_profile, user_data = calculate_scores()
    # Save to BigQuery with appending new data rather than replacing
    to_gbq(
        user_data,
        destination_table=f"{project_id}.{dataset_id}.movies_user_profiles_features",
        project_id=project_id,
        if_exists="append"
    )
    recommendations_similarity = get_recommendations(user_profile, numeric_data, data, 10, 7.0)
    print("\n\nTop 10 Movie Recommendations (Rating >= 7):")
    print(recommendations_similarity.to_string(index=False, justify='left'))