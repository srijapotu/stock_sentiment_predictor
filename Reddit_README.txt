
Reddit Sentiment Analysis

Overview
This script fetches posts from a specified subreddit using the Reddit API, processes the text for sentiment analysis, and trains a logistic regression model to classify the sentiment (positive, negative, or neutral). It outputs evaluation metrics and saves the processed data as a CSV file.

Requirements

Dependencies
- Python 3.7+
- Required Libraries:
  - praw
  - pandas
  - textblob
  - sklearn

You can install all dependencies using:
pip install -r requirements.txt

File Structure
- script.py: The main Python script.
- requirements.txt: Contains the required Python libraries.
- reddit_stock_market_data.csv: Output file containing fetched and processed data.

Setup Instructions

1. Clone or download this repository.
2. Install Python dependencies by running:
   pip install praw pandas textblob scikit-learn
3. Set up a Reddit API application:
   - Go to Reddit Developer.
   - Create an application and obtain the client_id, client_secret, username, and password.

4. Replace the placeholder credentials in the script with your Reddit API credentials:
   reddit = praw.Reddit(
       user_agent="your_app_name",
       client_id="your_client_id",
       client_secret="your_client_secret",
       username="your_username",
       password="your_password"
   )

Running the Script

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the script:
   python script.py
4. View the output in the console, and find the processed data saved in reddit_stock_market_data.csv.

Outputs
- Console Logs: Sentiment distribution, evaluation metrics, and a list of negative posts.
- CSV File: Processed subreddit data including sentiment labels.

Notes
- Replace the subreddit_name variable in the script with your desired subreddit.
- Adjust the limit parameter in the script to fetch more or fewer posts.

Example Commands
Install all dependencies:
pip install praw pandas textblob scikit-learn

Run the script:
python script.py

Check results in reddit_stock_market_data.csv.
