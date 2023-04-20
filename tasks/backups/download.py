
import os
import click
import subprocess
import requests
import zipfile
import random
import io
from tqdm import tqdm
import shutil
import gdown

ZIP_FILE = 'temp.zip'
FILE_ID = '1F2_jivRMH8cxBejRRthm3B14iRN0PMtS'
DATASETS_DIRECTORY = '.'

def main():
    print(f'downloading the file id: {FILE_ID}')
    download(FILE_ID, ZIP_FILE)

    print(f'unzipping the file into {DATASETS_DIRECTORY}')
    unzip(ZIP_FILE, DATASETS_DIRECTORY)

    print(f'removing temporary zip file {ZIP_FILE}')
    os.remove(ZIP_FILE)

def download(file_id, file_path):
    url = f'https://drive.google.com/uc?id={file_id}'

    gdown.download(url, file_path, quiet=False)


def unzip(file_path, extract_dir):
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        # Get the total number of files in the archive
        total_files = len(zip_file.infolist())

        # Use tqdm to create a progress bar
        progress_bar = tqdm(total=total_files,
                            desc='Extracting', unit=' files')

        # Extract all the files in the archive to the output directory
        zip_file.extractall(extract_dir)

        # Update the progress bar for each file that is extracted
        for _ in zip_file.infolist():
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

if __name__ == '__main__':
    main()
