import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from feedback.error_db import get_errors
from evaluation.metrics import evaluate_bleu, evaluate_rouge, evaluate_exact_match

# データ準備
def prepare_classifier_data():
    errors = get_errors()
    data = []
    for e in errors:
        question, pred, ref, error_type = e[1], e[2], e[3], e[4]
        bleu = evaluate_bleu(pred, ref)
        rouge1, rougeL = evaluate_rouge(pred, ref)
        em = evaluate_exact_match(pred, ref)
        data.append([question, pred, ref, bleu, rouge1, rougeL, em, error_type])
    df = pd.DataFrame(data, columns=["question", "prediction", "reference", "BLEU", "ROUGE-1", "ROUGE-L", "EM", "error_type"])
    return df

# モデル訓練
def train_classifier():
    df = prepare_classifier_data()
    X = df[["BLEU", "ROUGE-1", "ROUGE-L", "EM"]]
    y = df["error_type"].apply(lambda x: 1 if x == "incorrect" else 0)  # BAD=1, GOOD=0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # モデル保存
    with open("models/error_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    train_classifier()
