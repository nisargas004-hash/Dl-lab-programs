from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
reviews = [
    "I loved this movie, it was fantastic",
    "The product is excellent and very useful",
    "Amazing experience, I really enjoyed it",
    "The service was wonderful",
    "Very happy with my purchase",
    "This product is terrible",
    "I hated this movie",
    "Very disappointing experience",
    "The product broke after one day",
    "Poor quality and bad service"
]

# Labels: 1 = Positive, 0 = Negative
labels = [
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0
]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reviews,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict test data
y_pred = model.predict(X_test_tfidf)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Test new reviews
new_reviews = [
    "The movie was amazing and I loved it",
    "The product was awful and disappointing",
    "I am very happy with this purchase"
]

new_reviews_tfidf = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_tfidf)

# Display results
for review, prediction in zip(new_reviews, predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("\nReview:", review)
    print("Sentiment:", sentiment)
