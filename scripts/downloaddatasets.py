import os
import yaml
import requests
import argparse
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_file(url, dest_path):
    if not url:
        print(f"No URL provided for {dest_path}. Skipping.")
        return

    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"File already exists at {dest_path}. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {dest_path}.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for PACE pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--datasets", default="all", help="Comma-separated list of datasets to download (or 'all')")
    args = parser.parse_args()

    config = load_config(args.config)
    
    datasets_to_download = args.datasets.split(',')
    if 'all' in datasets_to_download:
        datasets_to_download = config['datasets'].keys()

    for dataset_name in datasets_to_download:
        dataset_name = dataset_name.strip()
        if dataset_name not in config['datasets']:
            print(f"Dataset {dataset_name} not found in config.")
            continue

        dataset_config = config['datasets'][dataset_name]
        url = dataset_config.get('source_url')
        raw_path = dataset_config.get('raw_path')
        
        if dataset_name == 'nij' and "ojp.usdoj.gov" in url:
             print(f"NOTE: NIJ dataset might require manual download or agreement. If download fails, please place the file manually at {raw_path}")

        download_file(url, raw_path)

if __name__ == "__main__":
    main()
