from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the dataset
dataset = pd.read_csv('data/processed/processed_data.csv')  # Update the path

# Load the TfidfVectorizer
with open('models/model.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Load the tfidf_matrix if saved separately
with open('models/tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix = pickle.load(file)

# Load tehe skills_matrix if saved separately
with open('models/skills_matrix.pkl', 'rb') as file:
    skills_matrix = pickle.load(file)

# Load the vectorizer if saved separately
with open('models/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    user_input = request.form['user_input']
    exact_match = dataset[dataset['Job Role Title'].str.lower() == user_input.lower()]
    
    if not exact_match.empty:
        skills_list = exact_match['Skills Required'].values[0]
        result = f"Exact match found!\nJob Role: {exact_match['Job Role Title'].values[0]}\nSkills: {skills_list}"
    else:
        user_input_vector = tfidf_vectorizer.transform([user_input])
        similarities = cosine_similarity(user_input_vector, tfidf_matrix)
        threshold = 0.2

        similar_roles = dataset[similarities[0] > threshold]

        if not similar_roles.empty:
            result = "Similar job roles found:\n"
            for _, row in similar_roles.head(3).iterrows():
                skills_list = row['Skills Required']
                result += f"Job Role: {row['Job Role Title']}\nSkills: {skills_list} \n"
        else:
            result = "No matching job roles found."
    return render_template('index.html', result=result)

@app.route('/process_skills', methods=['POST'])
def process_skills():
    user_skills = request.form['user_skills']
    # exact_match = dataset[dataset['Job Role Title'].str.lower() == user_input.lower()]
    # Check for an exact match
    exact_match = dataset[dataset['Job Role Title'].str.lower() == user_skills.lower()]

    if not exact_match.empty:
        suitable_job = f"Exact match found!\nJob Role: {exact_match.iloc[0]['Job Role Title']}"
    else:
        # If no exact match, find the best match based on word similarity
        user_input_vector = vectorizer.transform([user_skills])

        similarities = cosine_similarity(user_input_vector, skills_matrix)

        best_match_index = similarities.argmax()
        best_match = dataset.iloc[best_match_index]['Job Role Title']

        if best_match:
            suitable_job = f"Best match found!\nJob Role: {best_match}"
        else:
            suitable_job = "No matching job roles found."

    return render_template('index.html', suitable_job=suitable_job)

if __name__ == '__main__':
    app.run(debug=True)
