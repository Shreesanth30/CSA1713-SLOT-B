from textblob import TextBlob

# Get input from user
text = input("Enter a sentence: ")

# Perform sentiment analysis
blob = TextBlob(text)
polarity = blob.sentiment.polarity

# Display sentiment
if polarity > 0:
    print("Sentiment: Positive")
elif polarity < 0:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")

print("Polarity Score:", polarity)
