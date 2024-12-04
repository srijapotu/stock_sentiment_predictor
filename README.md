Telegram Sentiment Analysis
This project is a script for scraping messages from a Telegram
channel and performing sentiment analysis and
predictive modeling using TextBlob and Logistic Regression.

## Prerequisites
Before running this script, ensure you have the following installed:
- Python 3.7 or above
- pip (Python package manager)

## Dependencies
Install the required libraries using pip:
```bash
pip install telethon scikit-learn pandas textblob
```

## Setup Instructions
1. **Telegram API Credentials**
To access Telegram's API, you need an API ID and API Hash. Obtain
these by:
- Logging into your Telegram account at [Telegram's API
Development Tools](https://my.telegram.org/auth).
- Creating a new application and noting down the `API ID` and `API
Hash`.

2. **Set Your API Credentials**
Replace the placeholders in the script with your actual `API ID` and
`API Hash`:
```python
api_id = 'your_api_id'
api_hash = 'your_api_hash'
```

3. **Provide the Telegram Channel Link**
Replace the `channel_invite_link` variable in the script with the
invite link to your desired Telegram channel:
```python
channel_invite_link = 'https://t.me/your_channel_link'
```

## Running the Script
1. Save the script as `telegram_sentiment_analysis.py`.
2. Run the script using the following command:
```bash
python telegram_sentiment_analysis.py
```

## Script Overview
1. **Telegram Message Scraping**
The script uses the `Telethon` library to scrape the last 200
messages from the specified Telegram channel.

2. **Sentiment Analysis**
Messages are analyzed using `TextBlob` to assign a sentiment
polarity score:
- Positive sentiment: Polarity > 0
- Neutral sentiment: Polarity = 0 (filtered out in the cleaning step)
- Negative sentiment: Polarity < 0

3. **Predictive Model**
A Logistic Regression model is built to classify messages into
positive and negative sentiments:
- Features: Extracted using `CountVectorizer`.
- Evaluation: Outputs the model's accuracy, precision, and recall.

## Output
1. Sentiment scores for individual messages are printed to the
console.
2. The script prints the performance metrics for the predictive model:
- **Accuracy**
- **Precision**
- **Recall**

## Notes
- The script assumes the Telegram channel allows access to its
messages through the provided API credentials.
- Ensure compliance with Telegram's terms of service and privacy
policies when using the script.
