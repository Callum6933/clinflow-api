from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from clinflow.config import load_config
from clinflow.logging_utils import get_logger
from clinflow.data.load import load_dataset
from pathlib import Path


def train_model(df, cfg):
    # configure logging
    logger = get_logger(__name__)

    # create X (features) and y (target)
    y = df["target"]
    X = df.drop(["target", "num"], axis=1)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Test split created")

    # standardise/scale and transform
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    logger.info("Set scaled and transformed")

    # train logistic regression
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    logger.info("Logistic regression model trained")

    # predict
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # accuracy score
    roc_auc = roc_auc_score(y_test, y_pred)  # roc score
    logger.info("Accuracy scores calculated")

    # create dict with model, scaler and metrics
    model = {
        "scaler": scaler,
        "model": clf,
        "accuracy_score": accuracy,
        "roc_auc_score": roc_auc,
    }

    return model


def main():
    # configure logger
    logger = get_logger(__name__)

    # get cfg
    cfg = load_config()

    # load clean dataset
    clean_data_path = (
        Path(cfg["paths"]["processed_data"]["folder"])
        / cfg["paths"]["processed_data"]["file"]
    )
    df = load_dataset(clean_data_path)

    # train model
    model = train_model(df, cfg)
    logger.info("Model training successful")

    # print metrics
    print(f"Accuracy score: {model['accuracy_score']}")
    print(f"AUC score: {model['roc_auc_score']}")


if __name__ == "__main__":
    main()
