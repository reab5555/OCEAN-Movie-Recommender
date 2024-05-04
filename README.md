# OCEAN-Movie-Recommender
This algorithm is able to recommend movies based on the user's personality traits.
   
## Description
This algorithm recommends to the user the movies that it thinks are the most suitable based on the user's personality traits according to the Big Five theory.   
   
The Big Five or OCEAN model is a framework in psychology that identifies five broad personality traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. These traits are used to describe human personality and predict behavior.
   
By evaluating these traits in the user, the algorithm is able to predict the movies that best suit their personality type. The different levels of each trait coalesce into a certain personality type.
   
## Methodology
We have a dataset of movies and aggregated averages of the personality traits for each movie. The MyPersonality dataset derived from a Facebook app comprises personality scores of user's that liked these movies based on the Big Five traits, along with demographic and profile data from users who consented to share their information for research. The dataset we have contains a list of about 850 movies that facebook user's liked and their aggregated average measures in terms of each personality trait, including age and gender (Data on each user is not currently available).
   
The algorithm loads and preprocesses the dataset of movie attributes and user personality traits (we use Google Bigquery warehouse to store the data). It then computes movie recommendations based on the cosine similarity between a user's personality traits and the movies attributes like their average aggregated traits measures, gender, age, and unique keywords.
   
Finally, a GUI application simulation that interacts with the dataset manage user movie preferences based on their Big Five personality traits. It allows a user to log in with their ID to load and select their liked movies. User selections are used to update personality trait data in the dataset, potentially altering the dataset for future recommendations.





