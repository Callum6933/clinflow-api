import yaml
from pathlib import Path


def load_config(filepath=None):
    if filepath is None:
        current_file = Path(__file__)
        root = current_file.parent.parent.parent
        path = root / "config" / "config.yml"
    else:
        path = Path(filepath)

    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parsed_config = load_config()
    print(parsed_config)


if __name__ == "__main__":
    main()
