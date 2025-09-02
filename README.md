🚀 Project Overview
This project focuses on detecting potential hate speech directed toward the LGBTQIA+ community. It uses a supervised machine learning approach, where the model is trained on labeled data to learn patterns of hate speech vs. non-hate speech and then applies that knowledge to new, unseen text.

🛠️ Key Features
Text Classification: Identifies whether a given text contains hate speech or not.
Multiple Input Options:
User can directly enter text.
Upload a CSV file for bulk predictions.
Scrape tweets using ntscraper.
Fetch YouTube comments using the YouTube API.
Interactive Web App: Built with Streamlit for an easy-to-use interface.

🔎 How It Works
1. Data & Preprocessing:
A dataset containing LGBTQIA+-related hate speech and non-hate speech was used for training.
Preprocessing steps included:
Removing URLs, stopwords, punctuation, and digits
Tokenization
Lemmatization (reducing words to their base form)
Word frequency analysis was visualized using WordClouds to highlight common terms.

2. Feature Extraction
Used TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical vectors.
TF-IDF assigns higher importance to words that are frequent in a document but less common across the dataset, helping capture meaningful patterns.

3. Model Training
Implemented a Random Forest Classifier, an ensemble method that builds multiple decision trees and combines their predictions for better accuracy and robustness.

4. Deployment
The model is deployed as a Streamlit web app, making it accessible and interactive for end users.

📚 Tech Stack
Python
Natural Language Processing (NLP) → Tokenization, Lemmatization, Stopword Removal
TF-IDF → Feature extraction
Random Forest Classifier → Machine Learning model
Streamlit → User interface
ntscraper → Twitter scraping
YouTube API → Fetching comments
Screenshots of the output :
<img width="1683" height="852" alt="Screenshot 2025-09-02 233619" src="https://github.com/user-attachments/assets/bcf7a85d-a8b6-4b77-b5b1-552b7ac28c50" />

<img width="1732" height="836" alt="Screenshot 2025-09-02 233956" src="https://github.com/user-attachments/assets/c55edbc0-3d0e-488f-9507-8a73cffd3fed" />

<img width="1740" height="1033" alt="Screenshot 2025-09-02 234426" src="https://github.com/user-attachments/assets/9fdf1212-e4b7-4b03-b570-38bc5fa964e7" />

<img width="1702" height="783" alt="Screenshot 2025-09-03 001020" src="https://github.com/user-attachments/assets/d38ec9ba-d31d-4efd-b5a0-f132ee0b98f5" />

<img width="1892" height="893" alt="Screenshot 2025-09-03 003341" src="https://github.com/user-attachments/assets/a419ccf8-7566-4efe-a66f-7a183d887a8e" />

<img width="919" height="366" alt="Screenshot 2025-09-03 003953" src="https://github.com/user-attachments/assets/c912b31e-ee0e-4936-8ec8-6b7232b90fec" />








