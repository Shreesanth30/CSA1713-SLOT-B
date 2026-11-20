positive_words = ["good", "great", "excellent", "happy", "love",
                  "awesome", "wonderful", "best", "amazing"]

negative_words = ["bad", "worst", "poor", "sad", "hate",
                  "terrible", "awful", "disappointing"]

text = input("Enter a sentence: ")

words = text.lower().split()

positive = 0
negative = 0

for word in words:
    if word in positive_words:
        positive += 1
    elif word in negative_words:
        negative += 1

print("\nSentiment Analysis Result")

if positive > negative:
    print("Positive Sentiment")
elif negative > positive:
    print("Negative Sentiment")
else:
    print("Neutral Sentiment")

print("Positive words:", positive)
print("Negative words:", negative)
