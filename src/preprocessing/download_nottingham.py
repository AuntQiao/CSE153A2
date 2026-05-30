import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "http://ifdo.ca/~seymour/nottingham/nottingham_database.zip"


def download_nottingham(output_dir="data/raw/nottingham", url=DEFAULT_URL):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "nottingham.zip"

    if not zip_path.exists():
        print(f"Downloading Nottingham dataset from {url}")
        urlretrieve(url, zip_path)
    else:
        print(f"Using existing archive: {zip_path}")

    extract_dir = output_dir / "abc"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    print(f"Extracted ABC files to {extract_dir}")
    return extract_dir


def main():
    parser = argparse.ArgumentParser(description="Download the Nottingham ABC dataset.")
    parser.add_argument("--output-dir", default="data/raw/nottingham")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    download_nottingham(args.output_dir, args.url)


if __name__ == "__main__":
    main()
