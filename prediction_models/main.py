import argparse

from pipeline.io import load_data
from pipeline.preprocessing import preprocess
from pipeline.inference import predict
from pipeline.retrain import retrain
from prediction_models.dummy_model import DummyModel


def main():
    """
    Run the command-line entry point for the incident classification pipeline.

    Supported modes are training, prediction, and retraining. The function
    parses command-line arguments, initialises the model, and dispatches the
    requested workflow.

    Command-line Args:
        --mode: Required. One of ``train``, ``predict``, or ``retrain``.
        --input: Path to the primary input CSV file.
        --reviewed: Path to the reviewed CSV file used in retraining mode.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True, choices=["train", "predict", "retrain"])
    parser.add_argument("--input", help="input CSV path")
    parser.add_argument("--reviewed", help="reviewed CSV path")

    args = parser.parse_args()

    model = DummyModel()

    if args.mode == "train":
        df = load_data(args.input, mode="train")
        df = preprocess(df)

        X = df["text"]
        y = df["predicted_energy_type"]  # just an example

        model.fit(X, y)
        print("[INFO] Training completed")

    elif args.mode == "predict":
        df = load_data(args.input, mode="predict")
        df = preprocess(df)

        df = predict(model, df)

        df.to_csv("ML_Classification.csv", index=False)
        print("[INFO] Prediction saved to output.csv")

    elif args.mode == "retrain":
        model = retrain(args.input, args.reviewed, model)
        print("[INFO] Retraining completed")


if __name__ == "__main__":
    main()