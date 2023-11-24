

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
from src.vis.helpers import mean_plus_minus_std, format_criterium
from src.utils.hydra import get_original_cwd_safe


@hydra.main(version_base=None, config_path='../../conf', config_name='linear_classifier_best_model_soups')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')
    print(f'fetching results for {cfg.train_samples} train samples')

    if cfg.checkpoint_dir is None:
        # filters
        filters_random = get_best_criterium_filters(
            criterium='random',
            train_samples=cfg.train_samples,
        )
        filters_cold_paws = get_best_criterium_filters(
            criterium='cold_paws',
            train_samples=cfg.train_samples,
        )
        filters_furthest = get_best_criterium_filters(
            criterium='furthest',
            train_samples=cfg.train_samples,
        )
        filters_closest = get_best_criterium_filters(
            criterium='closest',
            train_samples=cfg.train_samples,
        )
        filters_half_in_half = get_best_criterium_filters(
            criterium='half_in_half',
            train_samples=cfg.train_samples,
        )
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

        def transform_runs_cold_paws(df):
            if len(df) == 0:
                return pd.DataFrame()

            for i in range(len(REQUIRED_METRICS)):
                df[REQUIRED_METRICS_SHORT_NAME[i]] = \
                    df.apply(lambda r: r['summary'].get(REQUIRED_METRICS[i], None), axis=1)

            df['dataset'] = df.apply(lambda r: r['config']['dataset']['name'], axis=1)
            df['classes'] = df.apply(
                lambda r: r['config']['classes'], axis=1)
            df['criterium'] = 'cold_paws'
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
            get_runs('linear-classifier-soup', filters=filters_random),
            get_runs('linear-classifier-soup', filters=filters_furthest),
            get_runs('linear-classifier-soup', filters=filters_half_in_half),
            get_runs('linear-classifier-soup', filters=filters_closest),
            get_runs('linear-classifier-soup', filters=filters_fps),
        ], ignore_index=True)
        df_general_soups = transform_runs_general(df_general_soups)
        print(f'total runs: {len(df_general_soups)}')
        print()

        print(f'ingredients: fetching furthest, closest, furthest/closest, fps, and random runs')
        df_general_ingredients = pd.concat([
            get_runs('linear-classifier', filters=filters_random),
            get_runs('linear-classifier', filters=filters_furthest),
            get_runs('linear-classifier', filters=filters_half_in_half),
            get_runs('linear-classifier', filters=filters_closest),
            get_runs('linear-classifier', filters=filters_fps),
        ], ignore_index=True)
        df_general_ingredients = transform_runs_general(df_general_ingredients)
        print(f'total runs: {len(df_general_ingredients)}')
        print()

        print('soups: fetching all cold paws runs')
        df_cold_paws_soups = get_runs('linear-classifier-cold-paws-soup', filters=filters_cold_paws)
        df_cold_paws_soups = transform_runs_cold_paws(df_cold_paws_soups)
        print(f'total runs: {len(df_cold_paws_soups)}')
        print()

        print('ingredients: fetching all cold paws runs')
        df_cold_paws_ingredients = get_runs('linear-classifier-cold-paws', filters=filters_cold_paws)
        df_cold_paws_ingredients = transform_runs_cold_paws(df_cold_paws_ingredients)
        print(f'total runs: {len(df_cold_paws_ingredients)}')
        print()

        # saving the dataframes
        df_general_soups.to_pickle('df_general_soups.pickle')
        df_general_ingredients.to_pickle('df_general_ingredients.pickle')
        df_cold_paws_soups.to_pickle('df_cold_paws_soups.pickle')
        df_cold_paws_ingredients.to_pickle('df_cold_paws_ingredients.pickle')
    else:
        print(f'loading checkpoint from: {cfg.checkpoint_dir}')
        df_general_soups = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_general_soups.pickle'))
        df_general_ingredients = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_general_ingredients.pickle'))
        df_cold_paws_soups = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_cold_paws_soups.pickle'))
        df_cold_paws_ingredients = pd.read_pickle(os.path.join(get_original_cwd_safe(), cfg.checkpoint_dir, 'df_cold_paws_ingredients.pickle'))

    # viz
    learning_rate_to_marker = {
        1e-1: 's',
        1e-2: 'D',
        1e-3: 'd',
    }

    # creating the figure
    for ssl in ['simclr/v1']:
        # merging dataframes
        df_soups = pd.concat([df_general_soups, df_cold_paws_soups], ignore_index=True)
        df_ingredients = pd.concat([df_general_ingredients, df_cold_paws_ingredients], ignore_index=True)

        # filtering by ssl
        df_soups = df_soups[df_soups['ssl'] == ssl].drop('ssl', axis=1)
        df_ingredients = df_ingredients[df_ingredients['ssl'] == ssl].drop('ssl', axis=1)

        ##############
        # zoomed-out #
        ##############

        seeds = 5
        dodge = .3

        fig, axes = plt.subplots(nrows=len(CRITERIA_ORDER), ncols=len(DATASETS_ORDER))
        for i, criterium in enumerate(CRITERIA_ORDER):
            for j, dataset in enumerate(DATASETS_ORDER):
                ax = axes[i][j]

                if i == 0:
                    ax.set_title(dataset, pad=10.0)

                if j == 0:
                    ax.set_ylabel(
                        format_criterium(criterium), 
                        rotation=0,
                        horizontalalignment='right',
                        verticalalignment='center',
                        labelpad=6.0,
                    )

                ax.set_xticks([])
                ax.set_yticks([])
    
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(1.0)
                    # ax.spines[axis].set_color('blue')

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

                    for lr_i, lr in enumerate(learning_rates):
                        ax.scatter(seed - 0.1, values[lr_i], s=15, color='#bababa', marker=learning_rate_to_marker[lr])

                    # soup
                    value = df_soups[
                        (df_soups['criterium'] == criterium) &
                        (df_soups['dataset'] == dataset) &
                        (df_soups['seed'] == seed)
                    ]['F1-MACRO'].item()
                    ax.scatter(seed+0.1, value, s=50, color='black', marker='+', label=seed)

                    all_values.append(value)
                    all_values.extend(values)
                    # ax.axis('off')

                min_value = min(all_values)
                max_value = max(all_values)
                diff = max_value - min_value
                ax.set_ylim([min_value - diff*dodge, max_value + diff*dodge])
                ax.set_xlim([0 - 0.8, seeds - 1 + 0.8])

        plt.subplots_adjust(
            left=0.2, 
            right=0.8,
            top=0.9, 
            bottom=0.1, 

            wspace=0.25, 
            hspace=0.4,
        )

        plt.tight_layout()
        plt.savefig(f"zoomed_out_model_soups_{cfg.train_samples}_{ssl.replace('/', '_')}.pdf")
        plt.clf()

        #############
        # zoomed-in #
        #############

        seeds = 5
        dodge = .3
        criterium = 'fps'
        dataset = 'jurkat'

        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3.3))

        ax.set_title(f'{dataset}, {format_criterium(criterium, train_samples=cfg.train_samples)}', pad=10.0)

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
        diff = max_value - min_value
        ax.set_ylim([0.0, max_value + diff*0.2])
        ax.set_xlim([0 - 0.4, seeds - 1 + 0.4])

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            handles=reversed(handles), labels=reversed(labels),
            loc='lower left', 
            framealpha=1.0, 
            fancybox=False, edgecolor='black',
        )

        plt.grid(visible=True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"zoomed_in_model_soups_{cfg.train_samples}_{ssl.replace('/', '_')}.pdf")
        plt.clf()

    print(f'everything saved to: {os.getcwd()}')

if __name__ == '__main__':
    main()
