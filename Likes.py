import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QLabel, QMessageBox
from PyQt6.QtCore import Qt
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from pandas_gbq import to_gbq

# Initialize the BigQuery client
client = bigquery.Client()
project_id = "python-code-running"
dataset_id = "moviesets"


# ensure that users liked movies table exists (A copy of the original table for live updating)
def ensure_table_exists(client, project_id, dataset_id):
    table_id = f"{project_id}.{dataset_id}.movies_user_profiles_likes"
    try:
        client.get_table(table_id)  # This ensures the table exists
    except NotFound:
        # Create the table if it does not exist
        schema = [
            bigquery.SchemaField("user_id", "STRING"),
            bigquery.SchemaField("movie", "STRING"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)  # API request
        print(f"Table {table_id} created.")


# ensure that updated movies table exists
def ensure_updated_table_exists(client, project_id, dataset_id):
    source_table_id = f"{project_id}.{dataset_id}.movies_big_five_set"
    updated_table_id = f"{project_id}.{dataset_id}.movies_big_five_updated"

    try:
        client.get_table(updated_table_id)
    except NotFound:
        job = client.copy_table(source_table_id, updated_table_id)
        job.result()
        print(f"Table {updated_table_id} created by copying {source_table_id}.")


# Ensure required tables exist before starting the app
ensure_table_exists(client, project_id, dataset_id)
ensure_updated_table_exists(client, project_id, dataset_id)


class UserIDWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel("Enter your User ID to load movie preferences:")
        layout.addWidget(label)

        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText('Enter User Name')
        layout.addWidget(self.user_id_input)

        submit_button = QPushButton('Login')
        submit_button.clicked.connect(self.submit_user_id)
        layout.addWidget(submit_button)

        self.setLayout(layout)
        self.setWindowTitle('User Name')
        self.show()

    def submit_user_id(self):
        user_id = self.user_id_input.text().strip()
        if user_id:
            try:
                query = f"SELECT user_id FROM `{project_id}.{dataset_id}.movies_user_profiles_features` WHERE user_id = '{user_id}'"
                user_data = client.query(query).to_dataframe()
                if not user_data.empty:
                    query = f"SELECT * FROM `{project_id}.{dataset_id}.movies_big_five_set`"
                    data = client.query(query).to_dataframe()
                    self.movie_selection_window = MovieSelectionWindow(data, user_id)
                    self.movie_selection_window.show()
                    self.close()
                else:
                    QMessageBox.warning(self, 'Invalid User Name', 'The entered User Name does not exist.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'An error occurred: {str(e)}')


class MovieSelectionWindow(QWidget):
    def __init__(self, movies, user_id):
        super().__init__()
        self.movies = movies
        self.user_id = user_id
        self.previous_selections = self.load_previous_selections()
        self.initUI()

    def load_previous_selections(self):
        previous_selections = []
        try:
            query = f"SELECT movie FROM `{project_id}.{dataset_id}.movies_user_profiles_likes` WHERE user_id = '{self.user_id}'"
            user_movies = client.query(query).to_dataframe()
            previous_selections = set(user_movies['movie'].tolist())
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while loading previous selections: {str(e)}')
        return previous_selections

    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel("Select the movies you like:")
        layout.addWidget(label)

        self.movie_list = QListWidget()
        for movie in self.movies['movie'].tolist():
            item = QListWidgetItem(movie)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if movie in self.previous_selections else Qt.CheckState.Unchecked)
            self.movie_list.addItem(item)

        layout.addWidget(self.movie_list)

        confirm_button = QPushButton('Confirm Selection')
        confirm_button.clicked.connect(self.confirm_selection)
        layout.addWidget(confirm_button)

        self.setLayout(layout)
        self.setWindowTitle('Movie Selection')
        self.show()

    def confirm_selection(self):
        current_selected_movies = {
            self.movie_list.item(i).text() for i in range(self.movie_list.count()) if self.movie_list.item(i).checkState() == Qt.CheckState.Checked
        }
        deselected_movies = self.previous_selections - current_selected_movies
        newly_selected_movies = current_selected_movies - self.previous_selections

        if current_selected_movies != self.previous_selections:
            self.update_movie_traits(newly_selected_movies, deselected_movies)
            self.save_selections(current_selected_movies)
        QMessageBox.information(self, 'Selection Confirmed', 'Your movie preferences have been updated.')
        self.close()

    def update_movie_traits(self, newly_selected_movies, deselected_movies):
        try:
            # Fetch user profile data for the current user to update traits for newly selected movies
            user_profile_query = f"SELECT * FROM `{project_id}.{dataset_id}.movies_user_profiles_features` WHERE user_id = '{self.user_id}'"
            user_profile = client.query(user_profile_query).to_dataframe()
            if user_profile.empty:
                QMessageBox.warning(self, 'Error', 'User profile not found.')
                return

            # Assumed total number of users contributing to the dataset
            total_users = 1000  # Placeholder for actual user count that might be dynamically fetched or estimated

            # Adjust traits for newly selected movies
            for movie in newly_selected_movies:
                movie_index = self.movies[self.movies['movie'] == movie].index[0]
                for trait in ['ope', 'con', 'ext', 'agr', 'neu', 'age', 'gender']:
                    original_trait_value = self.movies.at[movie_index, trait]
                    user_trait_value = user_profile.at[0, trait]
                    updated_trait_value = (total_users * original_trait_value + user_trait_value) / (total_users + 1)
                    self.movies.at[movie_index, trait] = updated_trait_value

            # For deselected movies, reverse the calculation or adjust accordingly
            for movie in deselected_movies:
                movie_index = self.movies[self.movies['movie'] == movie].index[0]
                for trait in ['ope', 'con', 'ext', 'agr', 'neu', 'age', 'gender']:
                    current_trait_value = self.movies.at[movie_index, trait]
                    user_trait_value = user_profile.at[0, trait]
                    # Assuming deselection should decrease the influence of this user's traits on the movie
                    updated_trait_value = (total_users * current_trait_value - user_trait_value) / (
                                total_users - 1) if total_users > 1 else self.movies.at[
                        movie_index, trait]  # Prevent division by zero
                    self.movies.at[movie_index, trait] = updated_trait_value

            # Update the `movies_big_five_updated` table in BigQuery with the new values
            for movie in newly_selected_movies.union(deselected_movies):
                update_query = f"""
                    UPDATE `{project_id}.{dataset_id}.movies_big_five_updated`
                    SET ope = {self.movies.at[movie_index, 'ope']},
                        con = {self.movies.at[movie_index, 'con']},
                        ext = {self.movies.at[movie_index, 'ext']},
                        agr = {self.movies.at[movie_index, 'agr']},
                        neu = {self.movies.at[movie_index, 'neu']},
                        age = {self.movies.at[movie_index, 'age']},
                        gender = {self.movies.at[movie_index, 'gender']}
                    WHERE movie = '{movie}'
                """
                client.query(update_query).result()

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred during traits update: {str(e)}')

    def save_selections(self, selected_movies):
        try:
            # Update the database with the current selections
            user_data = pd.DataFrame({'user_id': [self.user_id] * len(selected_movies), 'movie': list(selected_movies)})
            to_gbq(user_data, destination_table=f"{project_id}.{dataset_id}.movies_user_profiles_likes", project_id=project_id, if_exists="replace")
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while saving selections: {str(e)}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    user_id_window = UserIDWindow()
    sys.exit(app.exec())
