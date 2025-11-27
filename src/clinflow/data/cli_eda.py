from clinflow.data.load import load_raw_data
import argparse


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the dataset CSV file", default=None)
    args = parser.parse_args()
    return args.path


def print_metrics(path):
    # records/features count
    df = load_raw_data(path)
    print(f"Records: {df.shape[0]}")
    print(f"Features: {df.shape[1]}")

    # columns with (and no. of) missing data
    print(f"Missing values:\n{df.isnull().sum()}")

    # target distribution
    # - number of individuals with and without heart disease
    target_distribution = df["num"].value_counts()
    print(f"Without heart disease: {target_distribution[0]}")
    print(f"With heart disease: {target_distribution[1:].sum()}")

    # - percentage of individuals with each severity of heart disease
    print(
        f"Percentages:\nNo heart disease: {round((target_distribution[0]/target_distribution.sum()), 2) * 100}%"
    )
    print(
        f"Severity 1: {round((target_distribution[1]/target_distribution.sum()), 2) * 100}%"
    )
    print(
        f"Severity 2: {round((target_distribution[2]/target_distribution.sum()), 2) * 100}%"
    )
    print(
        f"Severity 3: {round((target_distribution[3]/target_distribution.sum()), 2) * 100}%"
    )
    print(
        f"Severity 4: {round((target_distribution[4]/target_distribution.sum()), 2) * 100}%"
    )


def main():
    # parse command line arguments
    file_path = parse_cli_args()

    # load dataset and print metrics
    print_metrics(file_path)


if __name__ == "__main__":
    main()
