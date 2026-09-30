"""
Generic maximum-entropy named-entity tagger.

Thesis reference: Section 4.2.2.3 "Amharic named entity recognition ...
We develop this system using a maximum entropy model which is a better
statistical model in NLP linguistic classification." A maximum entropy
classifier for multi-class problems is mathematically a multinomial
logistic regression model (identical decision function, fit by maximizing
conditional likelihood, which is why "MaxEnt" and "multinomial logistic
regression" are used interchangeably in the NLP literature). We use
scikit-learn's LogisticRegression(multi_class="multinomial") as a
faithful, dependency-light stand-in for the OpenNLP MaxEnt trainer the
original thesis used, since re-implementing a full GIS/L-BFGS MaxEnt
trainer from scratch would not change the underlying model family.

The same class is reused for:
  * Amharic NER (Section 4.2.2.3, item "Amharic named entity recognition")
  * English answer-type determination (Section 4.2.4.1 (2))
  * Amharic answer-type determination (same section)
by swapping in a different feature function and training set.
"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


class MaxEntTagger:
    """Token-sequence tagger: features(token_in_context) -> class label."""

    def __init__(self, feature_fn, default_label="O"):
        self.feature_fn = feature_fn
        self.default_label = default_label
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = LogisticRegression(max_iter=500)
        self.trained = False
        self.train_accuracy = None

    def train(self, tagged_sentences):
        """tagged_sentences: list[list[(token, label)]]"""
        X_feats, y = [], []
        for sent in tagged_sentences:
            tokens = [t for t, _ in sent]
            for i, (tok, label) in enumerate(sent):
                X_feats.append(self.feature_fn(tokens, i))
                y.append(label)
        if len(set(y)) < 2:
            # Degenerate corpus (only one class present) -- classifier
            # would be meaningless; fall back to majority-label tagging.
            self.trained = False
            self._majority_label = y[0] if y else self.default_label
            return
        X = self.vectorizer.fit_transform(X_feats)
        self.classifier.fit(X, y)
        self.train_accuracy = self.classifier.score(X, y)
        self.trained = True

    def tag(self, tokens):
        if not tokens:
            return []
        if not self.trained:
            return [getattr(self, "_majority_label", self.default_label) for _ in tokens]
        feats = [self.feature_fn(tokens, i) for i in range(len(tokens))]
        X = self.vectorizer.transform(feats)
        return list(self.classifier.predict(X))


class MaxEntClassifier:
    """Whole-instance classifier: features(instance) -> class label.

    Used for answer-type determination, where the 'instance' is a whole
    question rather than a token sequence (Section 4.2.4.1 (2)).
    """

    def __init__(self, feature_fn):
        self.feature_fn = feature_fn
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = LogisticRegression(max_iter=500)
        self.train_accuracy = None

    def train(self, instances, labels):
        X = self.vectorizer.fit_transform(self.feature_fn(inst) for inst in instances)
        self.classifier.fit(X, labels)
        self.train_accuracy = self.classifier.score(X, labels)

    def predict(self, instance):
        X = self.vectorizer.transform([self.feature_fn(instance)])
        return self.classifier.predict(X)[0]

    def predict_proba_dict(self, instance):
        X = self.vectorizer.transform([self.feature_fn(instance)])
        probs = self.classifier.predict_proba(X)[0]
        return dict(zip(self.classifier.classes_, probs))


def merge_bio_spans(tokens, labels, outside_label="O"):
    """Merge consecutive identical non-'O' labels into entity spans.

    Returns list of dicts: {text, label, start, end} with start/end being
    token indices (end exclusive), analogous to the span information
    Lucene term vectors provide for the position-based search described
    in Section 4.2.4.1 (3) / 4.2.6.
    """
    spans = []
    i = 0
    n = len(tokens)
    while i < n:
        if labels[i] == outside_label:
            i += 1
            continue
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        spans.append({
            "text": " ".join(tokens[i:j]),
            "label": labels[i],
            "start": i,
            "end": j,
        })
        i = j
    return spans
