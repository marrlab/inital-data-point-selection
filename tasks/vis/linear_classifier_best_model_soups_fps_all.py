

import os
import hydra
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from omegaconf import DictConfig, OmegaConf
from src.utils.utils import recursive_dict_compare
from src.utils.wandb import get_runs, dict_to_filters, get_best_criterium_filters
from src.models.build_data_path import get_vis_folder_path
from src.vis.constants import DATASETS_ORDER, CRITERIA_ORDER, REQUIRED_METRICS, REQUIRED_METRICS_SHORT_NAME 
from src.vis.helpers import mean_plus_minus_std, format_criterium, format_dataset
from src.utils.hydra import get_original_cwd_safe

plt.rcParams.update({'font.size': 17})

@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_best_model_soups')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')
    print(f'fetching results for {cfg.train_samples} train samples')

    if cfg.checkpoint_dir is None:
        filters_fps = get_best_criterium_filters(
            criterium='fps',
            train_samples=cfg.train_samples,
        )

        def transform_runs_general(df):
            if len(df) == 0:
                return pd.DataFrame()

            for i in range(len(REQUIRED_METRICS)):
                df[REQUIRED_METRICS_SHORT_NAME[i]] = \
                    df.apply(lambda r: r['summary'].get(REQUIRED_METRICS[i], None), axis=1)

            df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
            df['classes'] = df.apply(
                lambda r: r['config']['classes'], axis=1)
            df['criterium'] = df.apply(
                lambda r: r['config']['kmeans']['criterium'], axis=1)
            df['ssl'] = df.apply(
                lambda r: r['config']['training']['weights']['type'] + '/' + r['config']['training']['weights']['version'],
                axis=1
            ) 
            df['seed'] = df.apply(
                lambda r: r['config']['training']['seed'],
                axis=1
            ) 
            df['learning_rate'] = df.apply(
                lambda r: r['config']['training']['learning_rate'],
                axis=1
            ) 

            df = df.drop(['name', 'summary', 'config'], axis=1)
            df = df.dropna()

            return df

        print(f'soups: fetching furthest, closest, furthest/closest, fps, and random runs')
        df_general_soups = pd.concat([
            get_runs('linear-classifier-soup', filters=filters_fps),
        ], ignore_index=True)
        df_general_soups = transform_runs_general(df_general_soups)
        print(f'total runs: {len(df_general_soups)}')
        print()

        print(f'ingredients: fetching furthest, closest, furthest/closest, fps, and random runs')
        df_general_ingredients = pd.concat([
            get_runs('linear-classifier', filters=filters_fps),
        ], ignore_index=True)
        df_general_ingredients = transform_runs_general(df_general_ingredients)
        print(f'total runs: {len(df_general_ingredients)}')
        print()

        # saving the dataframes
        df_general_soups.to_pickle('df_general_soups.pickle')
        df_general_ingredients.to_pickle('df_general_ingredients.pickle')
    else:
        print(f'loading checkpoint from: {cfg.checkpoint_dir}')
        df_general_soups = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_general_soups.pickle'))
        df_general_ingredients = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_general_ingredients.pickle'))

    # viz
    learning_rate_to_marker = {
        1e-1: 's',
        1e-2: 'D',
        1e-3: 'd',
    }

    # creating the figure
    for ssl in ['simclr/v1']:
        # merging dataframes
        df_soups = pd.concat([df_general_soups], ignore_index=True)
        df_ingredients = pd.concat([df_general_ingredients], ignore_index=True)

        # filtering by ssl
        df_soups = df_soups[df_soups['ssl'] == ssl].drop('ssl', axis=1)
        df_ingredients = df_ingredients[df_ingredients['ssl'] == ssl].drop('ssl', axis=1)

        #############
        # zoomed-in #
        #############

        seeds = 5
        dodge = .3
        criterium = 'fps'

        for dataset in DATASETS_ORDER:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3.3))

            ax.set_title(format_dataset(dataset), pad=10.0)

            if dataset == DATASETS_ORDER[0]:
                ax.set_ylabel('F1-MACRO')

            ax.set_xticks(list(range(seeds)))
            ax.tick_params(axis=u'both', which=u'both', length=0)
            ax.set_xlabel('random seed')
            
            all_values = []
            for seed in range(seeds):
                # ingredients
                values = df_ingredients[
                    (df_ingredients['criterium'] == criterium) &
                    (df_ingredients['dataset'] == dataset) &
                    (df_ingredients['seed'] == seed)
                ]['F1-MACRO'].values
                learning_rates = df_ingredients[
                    (df_ingredients['criterium'] == criterium) &
                    (df_ingredients['dataset'] == dataset) &
                    (df_ingredients['seed'] == seed)
                ]['learning_rate'].values

                labels = [f'lr={lr}' for lr in learning_rates]
                for lr_i, lr in enumerate(learning_rates):
                    ax.scatter(seed - 0.1, values[lr_i], s=50, color='#bababa', marker=learning_rate_to_marker[lr], label=labels[lr_i] if seed == 0 else None)

                # soup
                value = df_soups[
                    (df_soups['criterium'] == criterium) &
                    (df_soups['dataset'] == dataset) &
                    (df_soups['seed'] == seed)
                ]['F1-MACRO'].item()
                ax.scatter(seed+0.1, value, s=150, color='black', marker='+', label='model soup' if seed == 0 else None)

                all_values.append(value)
                all_values.extend(values)

            min_value = min(all_values)
            max_value = max(all_values)

            # y_min = np.floor(10 * min_value) / 10
            y_min = 0.0
            y_max = np.ceil(10 * (max_value + 0.05)) / 10
            if dataset == 'retinopathy':
                y_min = 0.4

            y_step = 0.1
            ax.set_ylim((y_min, y_max))
            ax.set_yticks(np.arange(y_min, y_max + y_step, y_step))

            ax.set_xlim([0 - 0.4, seeds - 1 + 0.4])

            if dataset == DATASETS_ORDER[-1]:
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(
                    handles=reversed(handles), labels=reversed(labels),
                    loc='lower left', 
                    # ncols=len(handles),
                    framealpha=1.0, 
                    fancybox=False, edgecolor='black',
                )

            plt.grid(visible=True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"zoomed_in_model_soups_{dataset}_{cfg.train_samples}_{ssl.replace('/', '_')}.pdf")
            plt.clf()

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
