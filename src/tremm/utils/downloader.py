import logging
from pathlib import Path
import zipfile
import sys
import gdown
import os
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def extract_data(destination: Path) -> None:
    temp_zip_path = destination / "tmp.zip"
    
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    os.remove(temp_zip_path)
    
    macosx_folder = destination / "__MACOSX"
    if macosx_folder.exists() and macosx_folder.is_dir():
        shutil.rmtree(macosx_folder)
        logging.info("Removed __MACOSX folder.")

    extracted_folder = destination / "MendelianData"
    if extracted_folder.exists() and extracted_folder.is_dir():
        for item in extracted_folder.iterdir():
            target = destination / item.name
            item.rename(target)
        
        extracted_folder.rmdir()
    
    logging.info("Extraction completed.")
    logging.info(f"Files in {destination}: {os.listdir(destination)}")

def download_data(file_id: str, destination: Path):
    try:
        destination.mkdir(parents=True, exist_ok=True)
        temp_zip_path = destination / "tmp.zip"
        remote_file_url = f"https://drive.google.com/uc?id={file_id}"
        logging.info(f"Starting download from {remote_file_url} to {temp_zip_path}")
        gdown.download(remote_file_url,
                       str(temp_zip_path),
                       quiet=False
                    )
        logging.info(f"Download completed. Extracting {temp_zip_path} to {destination}")
        extract_data(destination=destination)
    except Exception as e:
        logging.error(f"An error occurred during the download process: {e}")
        raise
