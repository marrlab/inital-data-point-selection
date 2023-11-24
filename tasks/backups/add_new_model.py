
import re
import os
import click
import json
import shutil

MODELS_BACKUP_FOLDER = 'src/models/data'
MODEL_FILE_NAME = 'model.ckpt'

@click.command()
@click.option('--path', '-p', help='Path to the model checkpoint to add.', type=str)
@click.option('--type', '-t', help='Weights type: (simclr, swav, dino).', type=click.Choice(['simclr', 'swav', 'dino']))
@click.option('--version', '-v', help='Weights version, specifying the subfolder.', type=str)
@click.option('--dataset', '-d', help='Dataset name, specifying the top folder', type=click.Choice(['cifar10', 'matek', 'isic', 'retinopathy', 'jurkat']))
def main(path, type, version, dataset):
    model_folder_dst = os.path.join(
        MODELS_BACKUP_FOLDER,
        dataset, type, version
    )
    print(f'saving model to \'{model_folder_dst}\'')

    if not os.path.exists(model_folder_dst):
        os.makedirs(model_folder_dst)

    shutil.copy(path, os.path.join(model_folder_dst, MODEL_FILE_NAME))
    
    run_path, wandb_id = re.search(r'^(.+)/lightning_logs/([^/]+)/', path).groups()
    meta = {
        'run_path': run_path,
        'wandb_id': wandb_id,
    }

    with open(os.path.join(model_folder_dst, 'meta.json'), 'w') as file:
        json.dump(meta, file, indent=4)

if __name__ == '__main__':
    main()
