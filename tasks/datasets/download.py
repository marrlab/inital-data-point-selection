
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

DATASETS_DIRECTORY = 'src/datasets/data'
ZIP_FILE = 'temp.zip'
DATASET_FILE_IDS = {
    'isic': '1-WH0Pwqj5mOss1WnAk84HwV7iMJiDwYA',
    'isic_small': '1-edKs6IO5x2zsWSR9kaek1lWl3eqGns7',
    'isic_smallest': '1-Zy8F7GsUNmF7VG8nNQ7yBaAnZ7TCZ2Q',
    'isic_smallest_unbalanced': '1-UHuGRW0kTHlPfiFQ7kFdqV1FmyYZtdy',
    'retinopathy': '1-ezwo8E0OqNxk7VE69VROHT4nKGBidJH',
    'matek': '1-pIwf4OHz-g7_C0K9O0BHXGMc8_kze_z',
    'jurkat': '1-qTLskAJ_nIwgX-OLgs6sBdJoDpx8S9N',
    'cifar10': '1-rxN7fK9BL_0dOouqc_D50d6NmcvGil2'
}


@click.command()
@click.option('--datasets', '-d', multiple=True, type=click.Choice(list(DATASET_FILE_IDS.keys())),
              help='Name(s) of dataset(s) to download')
@click.option('--redownload', '-r', is_flag=True, default=False,
              help='Re-download file(s) even if they already exist in the output directory')
def main(datasets, redownload):
    """
    Download one or more files from a web link based on their names.
    """
    create_directory(DATASETS_DIRECTORY)

    for dataset in datasets:
        extract_dir = os.path.join(DATASETS_DIRECTORY, dataset)
        if not redownload and os.path.exists(extract_dir):
            click.echo(f"{dataset} already exists in. Skipping...")
            continue

        file_id = DATASET_FILE_IDS[dataset]

        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)

        download_and_unzip(file_id)

        click.echo(f"{dataset} has been downloaded to {extract_dir}.")


def download_and_unzip(file_id):
    print(f'downloading the file id: {file_id}')
    download(file_id, ZIP_FILE)

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


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully!")
    else:
        print(f"Directory '{directory}' already exists!")


if __name__ == '__main__':
    main()
