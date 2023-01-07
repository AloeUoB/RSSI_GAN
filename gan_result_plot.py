import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev
import csv
from GAN.gan_classification import uniq_count


def load_csv(filename, house_name):
    csv = pd.read_csv(filename, usecols=['runs', 'f1_macro', 'experiment'])
    csv['House'] = house_name
    return csv

def get_result_house(house_all, gan_option, num_epochs_all, num_epochs_map, shot, run_number, clf):
    for i in range(1, 4):
        house_name = house_all[i]
        if gan_option == 3:
            num_epochs = num_epochs_map[str(shot) + house_name]
        else:
            num_epochs = num_epochs_all[i]

        if gan_option == 1:
            model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)

        if gan_option == 2:
            model_name = "ConGAN_wgp_Trans_house_" + house_name + 'gen'
            filename = "GAN/result_classification/result_csv/" + clf+"_" + model_name + '_epoch' + str(num_epochs) + ".csv"

        if gan_option == 3:
            model_name = "ConGAN_wgp_Transhot" + str(shot) + "_house_" + house_name + "_run_" + str(run_number) + 'gen'
            filename = "GAN/result_classification/result_csv/"+ clf+"_" + model_name + '_epoch' + str(num_epochs) + ".csv"

        if i == 1:
            df = load_csv(filename, house_name)
        else:
            df = df.append(load_csv(filename, house_name))

    return df

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))

    house_all = ['A', 'B', 'C', 'D']
    gan_option = 2
    clf = 'rf'
    if gan_option == 1:
        num_epochs_all = [620, 520, 500, 620]
        train_Bsize = 128
        reduce_ap = False
    if gan_option == 2:
        num_epochs_all = [0, 1000, 800, 840]
        shot = None
        train_Bsize = 128
        num_epochs_map = None
    if gan_option == 3:
        num_epochs_all = None
        shot_all = [10, 5, 3, 1]
        run_map = {10: 2, 5: 1, 3: 1, 1: 1}
        num_epochs_map = {"10B": 240, "10C": 1300, "10D": 300,
                          "5B": 400, "5C": 1240, "5D": 600,
                          "3B": 240, "3C": 1100, "3D": 520,
                          "1B": 120, "1C": 760, "1D": 240}

    if gan_option == 3:
        plot_title = 'Few-shot training'+' ('+clf+' model)'
        for shot in shot_all[0:4]:
            run_number = run_map[shot]
            df = get_result_house(house_all, gan_option, num_epochs_all, num_epochs_map, shot, run_number, clf)
            shot_map = {1:'one-shot', 3:'three-shot', 5:'five-shot', 10:'ten-shot'}
            df['shot'] = shot_map[shot]
            if shot != shot_all[0]:
                df_all = df_all.append(df)
            else:
                df_all = df

        exp_map = {'experiment': 'experiments', 'exp1': 'No augmentation', 'exp2': 'Transfer GAN', 'exp3': 'SMOTE'}
        df_all = df_all.replace({"experiment": exp_map})

        sns.set(font_scale=1.5)
        # sns.set(rc={'figure.figsize': (2, 1)})
        g = sns.catplot(x="House", y="f1_macro",
                    hue="experiment", col="shot",
                    data=df_all, kind="bar",
                    legend=False,
                    height=5, aspect=0.7
                    );
        g.fig.set_size_inches(20, 8)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(plot_title, fontsize=20)
        for ax in g.axes.ravel():
            # add annotations
            for c in ax.containers:
                labels = [f'{(v.get_height()):.2f}' for v in c]
                ax.bar_label(c, labels=labels,
                             label_type='edge',
                             padding=15, fontsize=10)
            ax.margins(y=0.2)

        plt.legend(bbox_to_anchor=(1.01, 1),
                   borderaxespad=0, title='Experiments',
                   fontsize=14)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()
    else:
        run_number = None
        plot_title = 'Full training dataset'+'('+clf+' model)'
        df = get_result_house(house_all, gan_option, num_epochs_all, num_epochs_map, shot, run_number, clf)

        exp_map = {'experiment':'experiments','exp1': 'No augmentation', 'exp2': 'Transfer GAN', 'exp3': 'SMOTE'}
        df = df.replace({"experiment": exp_map})

        sns.set(font_scale=1.5)
        g = sns.catplot(x="House", y="f1_macro",
                    hue="experiment",
                    data=df, kind="bar",
                    legend=False,
                    height=7, aspect=1.2
                    );
        g.fig.set_size_inches(15, 8)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(plot_title, fontsize=20)
        # ax = g.facet_axis(0, 0)
        for ax in g.axes.ravel():
            # add annotations
            for c in ax.containers:
                labels = [f'{(v.get_height()):.2f}' for v in c]
                ax.bar_label(c, labels=labels,
                             label_type='edge',
                             padding=30, fontsize=16)
            ax.margins(y=0.5)

        plt.legend(bbox_to_anchor=(1.01, 1),
                   borderaxespad=0, title='Experiments',
                   fontsize=20)
        plt.ylim(0, 100)
        plt.xlabel('House', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('f1_macro', fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()


    # "GAN/result_classification/result_csv/"+ "clf_noSMOTE" +model_name + '_epoch' + str(num_epochs) + ".csv"
    # "GAN/result_classification/result_csv/"+ "clf_" + model_name + '_epoch' + str(num_epochs) + ".csv"
    # "GAN/result_classification/result_csv/"+ "rf_noSMOTE" + model_name + '_epoch' + str(num_epochs) + ".csv"