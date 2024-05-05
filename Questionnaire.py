import random
from datetime import datetime
import pandas as pd
from google.cloud.exceptions import NotFound
from google.cloud import bigquery


# ensure that users features table exists
def ensure_table_exists(client, project_id, dataset_id):
    table_id = f"{project_id}.{dataset_id}.movies_user_profiles_features"
    try:
        client.get_table(table_id)  # This ensures the table exists
    except NotFound:
        # Define the schema for a BigQuery table
        schema = [
            {'name': 'user_creation_date', 'type': 'TIMESTAMP'},
            {'name': 'user_id', 'type': 'STRING'},
            {'name': 'user_name', 'type': 'STRING'},
            {'name': 'gender', 'type': 'INTEGER'},
            {'name': 'age', 'type': 'INTEGER'},
            {'name': 'ope', 'type': 'FLOAT'},
            {'name': 'con', 'type': 'FLOAT'},
            {'name': 'ext', 'type': 'FLOAT'},
            {'name': 'agr', 'type': 'FLOAT'},
            {'name': 'neu', 'type': 'FLOAT'},
        ]
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)  # API request
        print(f"Table {table_id} created.")

# Questionnaire framework
def calculate_scores():
    user_profiles = {}
    client = bigquery.Client()
    project_id = "python-code-running"
    dataset_id = "moviesets"
    table_id = "movies_user_profiles_features"

    ensure_table_exists(client, project_id, dataset_id)

    # Query to check existing usernames
    query = f"SELECT user_name FROM `{project_id}.{dataset_id}.{table_id}`"
    query_job = client.query(query)  # Use the BigQuery Client to execute the query
    results = query_job.result()  # Get the results of the query
    existing_usernames = set([row['user_name'] for row in results])  # Create a set of existing usernames

    print('***Welcome to the Personality Movie Recommendation***')
    user_name = input("Please enter your name: ")

    # Check if the username already exists
    while user_name in existing_usernames:
        print(f"The username '{user_name}' already exists. Please try a different name.")
        user_name = input("Please enter your name: ")

    age = input("Please enter your age: ")
    while not age.isdigit() or int(age) < 0:
        print("Invalid age. Please enter a non-negative integer.")
        age = input("Please enter your age: ")
    age = int(age)

    gender = input("Please enter your gender (male = 0, female = 1): ")
    while gender not in ['0', '1']:
        print("Invalid gender. Please enter '0' for male or '1' for female.")
        gender = input("Please enter your gender (male = 0, female = 1): ")
    gender = int(gender)  # Make sure to convert to int

    questions = [
        ("ext", "I am the life of the party.", False),
        ("ext", "I don't talk a lot.", True),
        ("ext", "I feel comfortable around people.", False),
        ("ext", "I keep in the background.", True),
        ("ext", "I start conversations.", False),
        ("ext", "I have little to say.", True),
        ("ext", "I talk to a lot of different people at parties.", False),
        ("ext", "I don't like to draw attention to myself.", True),
        ("ext", "I don't mind being the center of attention.", False),
        ("agr", "I feel little concern for others.", True),
        ("agr", "I am interested in people.", False),
        ("agr", "I insult people.", True),
        ("agr", "I sympathize with others' feelings.", False),
        ("agr", "I am not interested in other people's problems.", True),
        ("agr", "I have a soft heart.", False),
        ("agr", "I am not really interested in others.", True),
        ("agr", "I take time out for others.", False),
        ("agr", "I feel others' emotions.", False),
        ("con", "I am always prepared.", False),
        ("con", "I leave my belongings around.", True),
        ("con", "I pay attention to details.", False),
        ("con", "I make a mess of things.", True),
        ("con", "I get chores done right away.", False),
        ("con", "I often forget to put things back in their proper place.", True),
        ("con", "I like order.", False),
        ("con", "I shirk my duties.", True),
        ("con", "I follow a schedule.", False),
        ("neu", "I get stressed out easily.", False),
        ("neu", "I am relaxed most of the time.", True),
        ("neu", "I worry about things.", False),
        ("neu", "I seldom feel blue.", True),
        ("neu", "I am easily disturbed.", False),
        ("neu", "I get upset easily.", False),
        ("neu", "I change my mood a lot.", False),
        ("neu", "I have frequent mood swings.", False),
        ("neu", "I get irritated easily.", False),
        ("ope", "I have a rich vocabulary.", False),
        ("ope", "I have difficulty understanding abstract ideas.", True),
        ("ope", "I have a vivid imagination.", False),
        ("ope", "I am not interested in abstract ideas.", True),
        ("ope", "I have excellent ideas.", False),
        ("ope", "I do not have a good imagination.", True),
        ("ope", "I am quick to understand things.", False),
        ("ope", "I use difficult words.", True)
    ]

    random.shuffle(questions) # randomize the questions

    scores = {trait: 0 for trait, _, _ in questions}
    total_questions = {trait: 0 for trait in scores}

    options = {1: "Strongly Disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly Agree"}

    for index, (trait, statement, is_reversed) in enumerate(questions, start=1):
        response = input(f"{index}/{len(questions)}: {statement} (1-5): ")
        while not response.isdigit() or int(response) not in options:
            print("Invalid input. Please enter a number from 1 to 5.")
            response = input(f"{index}/{len(questions)}: {statement} (1-5): ")
        response = int(response)
        if is_reversed:
            response = 6 - response
        normalized_response = response - 3
        scores[trait] += normalized_response
        total_questions[trait] += 1

    for trait in scores:
        max_score = 2 * total_questions[trait]
        scores[trait] = scores[trait] / max_score

    current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    user_profile = [gender, age] + [scores[trait] for trait in ["ope", "con", "ext", "agr", "neu"]]
    user_id = ''.join([str(random.randint(0, 9)) for _ in range(6)])

    # prepare the dataframe with the user unique parameters and features
    user_data = pd.DataFrame({
        'user_creation_date': [current_datetime],
        'user_id': [user_id],
        'user_name': [user_name],
        'gender': [gender],
        'age': [age],
        **{trait: [scores[trait]] for trait in scores.keys()}
    })
    # Assuming 'user_creation_date' is currently a string representing datetime
    user_data['user_creation_date'] = pd.to_datetime(user_data['user_creation_date'])
    # Format the datetime to match BigQuery's TIMESTAMP format if not already
    user_data['user_creation_date'] = user_data['user_creation_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Convert data types explicitly
    user_data['user_creation_date'] = pd.to_datetime(user_data['user_creation_date'])
    user_data['user_id'] = user_data['user_id'].astype(str)
    user_data['user_name'] = user_data['user_name'].astype(str)
    user_data['gender'] = user_data['gender'].astype(int)
    user_data['age'] = user_data['age'].astype(int)

    return user_id, user_profile, user_data