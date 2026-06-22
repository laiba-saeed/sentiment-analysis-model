# Loads trained model
# Runs sentiment predictions

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./sentiment-model",
    tokenizer="./sentiment-model"
)

examples = [
    "This product is fantastic. I would buy it again.",
    "Terrible experience. Completely disappointed."
]

for text in examples:
    result = classifier(text)[0]

    sentiment = (
        "Positive"
        if result["label"] == "LABEL_1"
        else "Negative"
    )

    confidence = result["score"] * 100

    print("\nText:")
    print(text)

    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")