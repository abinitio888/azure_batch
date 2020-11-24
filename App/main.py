import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="App main entry")
    parser.add_argument("input_path", type=str, help="Job input path")
    parser.add_argument("output_path", type=str, help="Job output path")
    args = parser.parse_args()

    fd = open(args.input_path)
    df = pd.read_csv(fd, index_col=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    clf = RandomForestClassifier(n_estimators=10, max_depth=None,
            min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, X, y, cv=5)
    mean_score = scores.mean()
    with open(args.output_path, "w") as fd:
        fd.write(f"The model score: {mean_score}")
