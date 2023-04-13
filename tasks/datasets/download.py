
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
    'isic': '1bZe-IU-KtHLqKVXwng6_5vXf3SCZV3F6',
    'isic_small': '11UFaBtXM3I2JCsOhAhAx5eP5JNsiaDac',
    'isic_smallest': '13lllhxlnYgMgKzxZK-VeAnBYdyWyFMJy',
    'isic_smallest_unbalanced': '1aBTkvbiWzTVhzErqrPGWqqfUeK1U-PDI',
    'retinopathy': '1-2z6P8vGVWiFucD4lSnlFGEuyb2hTpCM',
    'matek': '1-9eEvnaWXedo2LSTCRS19V_jyZZiffEt',
    'jurkat': '1NYNJ-pv7XUgpkbgQVom1bpfZ_uc7n5ml',
    'cifar10': '1m2ardUoEybZ0d3hpcueEjFDwadTi4dG1'
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
