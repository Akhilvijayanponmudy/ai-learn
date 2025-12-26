from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import logging

logger = logging.getLogger(__name__)

class Classifier:
    def __init__(self):
        # Mock training data for baseline
        self.data = [
            ("my app is crashing", "bug"),
            ("cannot login to account", "bug"),
            ("error 500 when saving", "bug"),
            ("how do i reset password", "how-to"),
            ("where is the settings page", "how-to"),
            ("how to export data", "how-to"),
            ("billing info is wrong", "billing"),
            ("upgrade my plan", "billing"),
            ("invoice not received", "billing"),
            ("please add dark mode", "feature-request"),
            ("integration with slack would be nice", "feature-request"),
            ("can you support german language", "feature-request"),
        ]
        self.pipeline = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression()
        )
        self.train()

    def train(self):
        texts, labels = zip(*self.data)
        self.pipeline.fit(texts, labels)
        logger.info("Classifier trained on initial dataset")

    def predict(self, text: str) -> str:
        return self.pipeline.predict([text])[0]
