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
    
    # Extract all files from the ZIP to the destination folder.
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    os.remove(temp_zip_path)
    
    # Remove the __MACOSX folder if it exists.
    macosx_folder = destination / "__MACOSX"
    if macosx_folder.exists() and macosx_folder.is_dir():
        shutil.rmtree(macosx_folder)
        logging.info("Removed __MACOSX folder.")

    # Check if the extracted files are inside a folder named "MendelianData".
    extracted_folder = destination / "MendelianData"
    if extracted_folder.exists() and extracted_folder.is_dir():
        # Move each item from the MendelianData folder to the destination.
        for item in extracted_folder.iterdir():
            target = destination / item.name
            # If there's a naming conflict, handle it as needed.
            item.rename(target)
        
        # Remove the now empty "MendelianData" folder.
        extracted_folder.rmdir()
    
    logging.info("Extraction completed.")
    logging.info(f"Files in {destination}: {os.listdir(destination)}")

def download_data(file_id: str, destination: Path):
    try:
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