import praw
import pandas as pd
from textblob import TextBlob
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Reddit Authentication
reddit = praw.Reddit(
    user_agent="relationship_bot/0.1 by u/PersonalityPurple387",
    client_id="gkXiewe2kN2MIFqNRIjIWA",
    client_secret="pwbO2gMgGWmwGQFDvkRa3wClYpkalw",
    username="PersonalityPurple387",
    password="Abhinav@2005"
)

subreddit_name = "StockMarket"  # Replace if needed
subreddit = reddit.subreddit(subreddit_name)

# Fetching posts from the subreddit
posts = []
print("Fetching posts...")
for post in subreddit.hot(limit=200):  # Adjust the limit as needed
    print("Processing post:", post.title)  # Debug
    posts.append({
        "title": post.title,
        "text": post.selftext if post.selftext else "No Text",
        "score": post.score,
        "comments": post.num_comments,
    })

df = pd.DataFrame(posts)
print("Total posts fetched:", len(df))

# Clean Text
def clean_text(text):
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text.strip()

df["clean_text"] = df["text"].apply(lambda x: clean_text(x) if isinstance(x, str) else "")

# Sentiment Analysis
print("Performing sentiment analysis...")

# Calculate sentiment polarity
df["sentiment"] = df["clean_text"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Label sentiment based on polarity
def label_sentiment(polarity):
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

df["sentiment_label"] = df["sentiment"].apply(label_sentiment)

# Display sentiment distribution for verification
sentiment_counts = df["sentiment_label"].value_counts()
print("Sentiment distribution:")
print(sentiment_counts)

# Save to CSV
output_file = "reddit_stock_market_data.csv"
df.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")

# Encode sentiment labels for model input
label_encoder = LabelEncoder()
df["sentiment_encoded"] = label_encoder.fit_transform(df["sentiment_label"])

# Prepare data for machine learning
X = df["clean_text"]  # Features (text data)
y = df["sentiment_encoded"]  # Target (sentiment labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical form using Bag of Words (CountVectorizer)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Print negative posts
negative_posts = df[df["sentiment_label"] == "negative"]
print("\nNegative posts:")
for index, row in negative_posts.iterrows():
    print(f"- {row['title']}")

# Matplotlib Visualizations

# Sentiment Distribution Visualization (Positive, Negative, Neutral)
print("Visualizing sentiment distribution...")
sentiment_counts.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Frequency")
plt.show()

# Confusion Matrix Visualization
print("Visualizing confusion matrix...")
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = range(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_)
plt.yticks(tick_marks, label_encoder.classes_)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
