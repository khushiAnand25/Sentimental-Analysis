import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from nltk.corpus import stopwords
import spacy
import emoji
from ntscraper import Nitter
import pandas as pd
import googleapiclient.discovery
import pandas as pd


# Add a background image using custom CSS
st.markdown(
    """
    
    """,
    unsafe_allow_html=True
)

# Load the trained model
model = joblib.load("upload the model")

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load("upload the file pickel")

# Define preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Removing special symbols, hashtags, and other non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(words)

    # Handling chat words
    text = chat_convo(text)

    # Handling emojis
    text = demojize_text(text)

    # Tokenization using spaCy
    nlp = spacy.load("en_core_web_sm")
    tokens = [token.text for token in nlp(text)]

    return tokens

# Define chat conversation replacements
chat_words = {
    "lol": "laugh out loud", "brb": "be right back", "ttyl": "talk to you later",
    "gtg": "got to go", "btw": "by the way", "omg": "oh my god", "idk": "i don't know",
    "imho": "in my humble opinion", "icymi": "in case you missed it", "fyi": "for your information",
    "smh": "shaking my head", "rofl": "rolling on the floor laughing", "dm": "direct message",
    "dm me": "send me a direct message","imo": "in my opinion", "tbh": "to be honest",
    "dms open": "direct messages are open for communication", "lmao": "laughing my *a* off",
    "fwiw": "for what it's worth", "afk": "away from keyboard", "asap": "as soon as possible",
    "bff": "best friends forever", "faq": "frequently asked questions", "fyeo": "for your eyes only",
    "jk": "just kidding", "nvm": "never mind", "otoh": "on the other hand", "tmi": "too much information",
    "tyt": "take your time", "yolo": "you only live once", "wth": "what the heck", "roflmao": "rolling on the floor laughing my *a* off",
    "til": "today i learned", "ootd": "outfit of the day", "nsfw": "not safe for work", "oot": "out of town",
    "lolz": "laugh out loud, but with a 'z' for emphasis", "g2g": "got to go"
}

# Define function to replace chat words
def chat_convo(text):
    new_text = []
    for w in text.split():
        w = w.strip(string.punctuation)
        if w.lower() in chat_words:
            new_text.append(chat_words[w.lower()])
        else:
            new_text.append(w)
    return " ".join(new_text)

# Define function to handle emojis
def demojize_text(text):
    new_text = []
    new_text.append(emoji.demojize(text))
    return " ".join(new_text)

def main():
    # Sidebar with options
    st.sidebar.title("Options")
    option = st.sidebar.radio("Choose an option", ["Home", "Tweets Analysis", "YouTube Comments Analysis", "File Upload","About"])

    # Handle selected option
    if option == "Home":
        st.title("Detection of Hate Speech  on Social Media")
        st.write("To detect and predict potential textual hate speech targetting people")
        st.write("Enter text in the box to check if it is Hate Speech")
        user_input = st.text_area("Enter text here")
        if st.button("Check for Hate Text"):
            prediction = predict(user_input)
            if prediction == 0:
                st.write("**Unlikely to be Hate Speech**")
            else:
                st.write("**Hate Speech**")

    if option == "Tweets Analysis":
        st.title("Tweets Prediction for Textual Hate Speech")
        st.write("Enter term or a username to scrape tweets and predict for any potential hate speech")
        option = st.radio("Choose an option", ["Term", "Username"])
        if option == "Term":
            term = st.text_input("Enter term")
        elif option == "Username":
            username = st.text_input("Enter Twitter username")
        if st.button("Scrape Tweets"):
            if option == "Term":
                if term.strip() == "":
                    st.warning("Please enter a valid term.")
                else:
                    scraper = Nitter()
                    tweets = scraper.get_tweets(term, mode='term', number=10)
            elif option == "Username":
                if username.strip() == "":
                    st.warning("Please enter a valid Twitter username.")
                else:
                    scraper = Nitter()
                    tweets = scraper.get_tweets(username, mode='user', number=10)
            if tweets is not None:
                tweet_texts = [tweet['text'] for tweet in tweets['tweets']]
                predictions = []
                for text in tweet_texts:
                    prediction = predict(text)
                    predictions.append(prediction)
                df = pd.DataFrame({'Tweet': tweet_texts, 'Prediction': predictions})
                st.write("Predictions for Tweets:")
                st.write(df)
            else:
                st.error("Failed to retrieve tweets. Please check the username and try again.")

    elif option == "YouTube Comments Analysis":
        st.title("Youtube Comments Analysis")
        st.write("Enter a YouTube video link to scrape comments and make predictions")
        vid_input = st.text_area("Enter video link here")
        if st.button("Scrape comments"):
            if vid_input.strip() == "":
                st.warning("Please enter a valid link")
            else:
                dev = "(paste your youtube API key here)"  # YouTube API key
                api_service_name = "youtube"
                api_version = "v3"
                DEVELOPER_KEY = dev
                youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
                from langdetect import detect  # Importing the detect function from langdetect library

                def extract_comments(video, language='en'):
                    request = youtube.commentThreads().list(part="snippet", videoId=video, maxResults=100)
                    comments = []
                    response = request.execute()
                    count=0
                    f=0
                    while True:
                        for item in response['items']:
                            comment = item['snippet']['topLevelComment']['snippet']
                            public = item['snippet']['isPublic']
                            comments.append([comment['textOriginal']])
                            count=count+1
                            if(count>100):
                              f=1
                              break
                        if(f==1):
                            break
                        try:
                            nextPageToken = response['nextPageToken']
                        except KeyError:
                            break
                        nextRequest = youtube.commentThreads().list(part="snippet", videoId=video, maxResults=100,
                                                                    pageToken=nextPageToken)
                        response = nextRequest.execute()
                    st.write(comments)
                    # Filter comments by language
                    filtered_comments = []
                    prediction = []
                    for comment in comments:
                        try:
                            if detect(comment[0]) == language:
                                val = predict(comment[0])
                                filtered_comments.append(comment[0])
                                prediction.append(val)
                        except:
                            continue
                    df2 = pd.DataFrame({'Comment': filtered_comments, 'Prediction': prediction})
                    return df2

                df = extract_comments(vid_input)
                st.write(df)
    elif option=="File Upload":
        st.title("Upload files for Prediction")
        st.write("Predict hate text from data present present in a .csv file")
        st.write("Make sure that the data is present in 'text' column")
        uploaded_file = st.file_uploader("Choose a file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, usecols=['text'])  # Read only the 'text' column
            st.write("Preview of the 'text' column from the uploaded file:")
            st.write(df.head())
            # Make predictions from the uploaded CSV file
            predictions = []
            for index, row in df.iterrows():
                prediction = predict(row['text'])
                predictions.append(prediction)

            # Add predictions as a new column in the DataFrame
            df['Prediction'] = predictions

            # Display the DataFrame with predictions
            st.write("Predictions for uploaded data:")
            st.write(df)
    elif option=="About":
        st.title("Detection of Hate Speech  on Social Media")
        st.write("""
    It doesn't seem surprising that even today, there is a stigma around the LGBT+ community, and individuals often face bullying and shaming in their lives. Whether it's on social media platforms or other video streaming applications where they share content or express their thoughts, users may encounter hate speech and insensitive trolling in the comments section, which can significantly impact their morale. This application aims to provide a user-friendly interface for predicting hate speech, helping filter out offensive or hateful comments.

    **About the Project:**

    **Model Training:** I trained a machine learning model to detect hate against LGBTQIA+ individuals. Using a supervised learning approach, the model learned patterns from labeled data to make predictions on unseen text. We utilized a dataset containing examples of hate speech and non-hate texts related to LGBTQIA+ topics for model training.

    **Classification:** The trained model performs binary classification, distinguishing between hate text and non-hate text. Hate speech includes any communication that disparages LGBTQIA+ individuals based on their sexual orientation, gender identity, or other related attributes, while non-hate speech comprises text that does not exhibit characteristics of hate speech.

    **Text Preprocessing:** I preprocessed the data to retain essential text for training and testing while removing unnecessary elements. This included removing HTML tags, punctuation, stop words, converting text to lowercase, replacing emoticons with their textual meaning, handling special symbols, and tokenizing the data.

    **Vectorization:** Text data was converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. TF-IDF assigns weights to words based on their frequency in a document relative to their frequency in the entire corpus, helping capture the importance of words in a document.

    **Classifier Used:** Then ,I employed the Random Forest classifier, an ensemble learning method that builds multiple decision trees during training and combines their predictions to make a final prediction. Random Forest introduces randomness by selecting random subsets of the training data and features for each decision tree, reducing overfitting and improving the model's generalization performance.

    **Data Scraping:** To enable hate speech prediction from widely used platforms like Twitter and YouTube, we included data scraping functionality in this project. We utilized the Python module 'ntscraper' for scraping tweets based on a username or keyword and the YouTube API for scraping comments from a video link.

    **Conclusion:** This project aims to combat hate speech directed at the LGBTQIA+ community by providing a tool to identify and filter out offensive comments. By leveraging machine learning and data scraping techniques, we strive to create a safer and more inclusive online environment for all individuals, regardless of their sexual orientation or gender identity.
    """)



def predict(text):
    # Preprocess the input text
    tokens = preprocess_text(text)
    # Convert tokens back to text
    preprocessed_text = ' '.join(tokens)

    # Transform preprocessed text using the loaded TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([preprocessed_text])

    # Make predictions using the loaded model
    prediction = model.predict(text_vectorized)[0]

    return prediction

if __name__ == "__main__":
    main()