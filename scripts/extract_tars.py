import os
import tarfile
import argparse
from pathlib import Path


def main(src_path: str):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.tar'):
                tar_path = Path(root) / file
                extract_path = tar_path.parent
                
                print(f"Extracting {tar_path} to {extract_path}")
                try:
                    with tarfile.open(tar_path) as tar:
                        tar.extractall(path=extract_path)
                except Exception as e:
                    print(f"Error extracting {tar_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.src_path)
