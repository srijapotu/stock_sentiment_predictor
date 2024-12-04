# Install required libraries before running the script:
# pip install telethon scikit-learn pandas textblob

from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Step 1: Telegram API credentials and scraping
api_id = '24758574'  # Your API ID from Telegram
api_hash = 'a0e2ba8b2a8b3fe65253528070b0af1b'  # Your API Hash from Telegram

# The invite link of the Telegram channel
channel_invite_link = 'https://t.me/trade'  # Telegram channel's invite link

client = TelegramClient("session_name", api_id, api_hash)

def scrape_telegram(channel_invite_link, limit=200):
    with client:
        messages = []
        # Extract the channel username from the invite link
        channel_username = channel_invite_link.split('/')[-1]
        
        # Fetch message history from the channel
        history = client(GetHistoryRequest(
            peer=channel_username,
            limit=limit,
            offset_date=None,
            offset_id=0,
            max_id=0,
            min_id=0,
            add_offset=0,
            hash=0
        )).messages
        
        for message in history:
            if message.message:  # Only include text messages
                messages.append(message.message)
                
        return pd.DataFrame({"messages": messages})

# Connect to Telegram and scrape messages
client.start()
data = scrape_telegram(channel_invite_link, limit=200)

# Step 2: Data cleaning and sentiment analysis using TextBlob
def get_sentiment(text):
    # Use TextBlob to get the sentiment polarity
    blob = TextBlob(text)
    return blob.sentiment.polarity

data["sentiment"] = data["messages"].apply(get_sentiment)

# Print each message and its sentiment
for index, row in data.iterrows():
    print(f"Message: {row['messages']}")
    print(f"Sentiment: {row['sentiment']}
")

# Step 3: Data cleaning and preparing for the prediction model
data_cleaned = data.dropna()
data_cleaned = data_cleaned[data_cleaned["sentiment"] != 0]

# Step 4: Building and testing the prediction model
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(data_cleaned["messages"]).toarray()
y = (data_cleaned["sentiment"] > 0).astype(int)  # Positive sentiment = 1, Negative = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
