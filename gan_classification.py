import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import torch
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats


from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from general_utils import uniq_count
from GAN.gan_utils import load_GAN_model, generate_fake_data, get_fake_rssi,\
    MinMaxScaler, renormalization, windowing, get_mean_stat, get_stats,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_B, map_C, map_D, house_map,\
    binary_convert, num_room_house, train, test, load_house_data, get_col_use, \
    select_data_number
from play_space import feature, augment
from gan_evaluation import multiclass, rf_multiclass, classification_f1, rf_classification_f1
from imblearn.over_sampling import SMOTE
from gan_visual import  plot_all_room

def prep_data(X, y):
    X_prep = torch.from_numpy(np.transpose(X, (0, 2, 1)))
    y_prep = torch.from_numpy(y - 1).int()
    return X_prep, y_prep

def generate_dataframe(runs, all_f1, label):
    array_data = np.transpose(np.concatenate((np.array([runs]), np.array([all_f1]))))
    df = pd.DataFrame(array_data, columns=['runs', 'f1_macro'])
    df['experiment'] = label
    return df

def print_mean_f1(all_f1, exp):
            print('exp'+str(exp)+'-f1_mac:',sum(all_f1)/len(all_f1))


def duplicate(X, y, dup_number):
    dup_shot = X
    new_y = []
    for yi in range(0, len(y)):
        new_y.append(y[yi])

    for i in range (dup_number-1):
        dup_shot = np.vstack([dup_shot, X])
        for yi in range(0, len(y)):
            new_y.append(y[yi])

    return dup_shot, new_y

def smote_sampling(X_ori, y_ori, number_samp):
    X = X_ori
    y = y_ori
    X = X.reshape(X.shape[0], X_ori.shape[1] * X_ori.shape[2])
    # add extra label as a tag for number of sampling
    unique_y, counts_y = np.unique(y, return_counts=True)
    if counts_y[0] < 3:
        X = np.concatenate((X, X, X))
        y = np.concatenate((y, y, y))

    for n in range(number_samp):
        y = np.concatenate((y, [len(unique_y) + 1]))
        X = np.concatenate((X, np.zeros((1, X.shape[1]))))
    # perform SMOTE oversampling
    sm = SMOTE(k_neighbors=1)
    X_res, y_res = sm.fit_resample(X, y)
    # clean out the extra label tag
    X_res = X_res[y_res!=[len(unique_y) + 1]]
    y_res = y_res[y_res!=[len(unique_y) + 1]]
    # reshape back
    X_res = X_res.reshape(-1, X_ori.shape[1] * X_ori.shape[2])

    return X_res, y_res

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    house_all = ['A', 'B', 'C', 'D']
    shot_all = [10, 5, 3, 1]
    run_map = {10:4, 5:4, 3:4, 1:4}
    gan_option = 3
    if gan_option == 1:
        num_epochs_all = [620, 520, 500, 620]
        train_Bsize = 128
    if gan_option == 2:
        num_epochs_all = [0, 1000, 800, 840]
        train_Bsize = 128
    if gan_option == 3:
        shot_all = [10, 5, 3, 1]
        run_map = {10:4, 5:4, 3:4, 1:4}
        num_epochs_map = {"10B": 240, "10C": 1300, "10D": 300,
                          "5B": 400, "5C": 1240, "5D": 600,
                          "3B": 240, "3C": 1100, "3D": 520,
                          "1B": 120, "1C": 760, "1D": 240}
    confusion_matrix = False
    # exp1 = False
    # exp2 = False
    # exp3 = True
    runs = 10
    for shot in shot_all[0:1]:
        run_number = run_map[shot]
        for i in range(1,4):
            house_name = house_all[i]
            if gan_option ==3:
                num_epochs = num_epochs_map[str(shot) + house_name]
            else:
                num_epochs = num_epochs_all[i]  # A 620, B 520, C 440(reduce 160), D 620
            reduce_ap = False
            windowed_data, windowed_label, APs, NUM_CLASSES = \
                load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False)
            windowed_data_fl, windowed_label_fl, APs, NUM_CLASSES =\
                load_house_data(data_directory, house_name, datatype='fl', reduce_ap=False)
            if gan_option == 3:
                # select shot data
                X_shot, y_shot = select_data_number(windowed_data, windowed_label, shot, NUM_CLASSES)
                # prepare data
                X_train, y_train = prep_data(X_shot, y_shot)
                X_test, y_test = prep_data(windowed_data_fl, windowed_label_fl)
            else:
                X_train, y_train = prep_data(windowed_data, windowed_label)
                X_test, y_test = prep_data(windowed_data_fl, windowed_label_fl)

            WINDOW_SIZE = 20
            total_number = 1000
            low_bound = None
            if gan_option == 1:
                model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
            if gan_option == 2:
                model_name = "ConGAN_wgp_Trans_house_" + house_name + 'gen'
            if gan_option == 3:
                model_name = "ConGAN_wgp_Transhot" + str(shot) + "_house_" + house_name + "_run_" + str(
                    run_number) + 'gen'
                train_Bsize = shot*NUM_CLASSES

            print(model_name)
            epochs = 1200
            idx_runs = []
            for i in range(runs):
                idx_runs.append(i + 1)

            mlp = True
            if mlp:
                exp1 = False
                exp2 = False
                exp3 = True
                # experiment 1
                if exp1:
                    print("clf_experiment1: no augmentation")
                    all_f1_exp1 = []
                    for i in range(runs):
                        f1_exp1 = classification_f1(X_train, y_train, X_test, y_test,
                                                    APs, NUM_CLASSES, epochs, train_Bsize, device, False, exp=1,
                                                    flatten=True, show_epoch=False, confusion_met=confusion_matrix)
                        all_f1_exp1.append(f1_exp1)
                        print('exp1 run', i, 'is done')
                    df_real = generate_dataframe(idx_runs, all_f1_exp1, 'exp1')
                    print_mean_f1(all_f1_exp1, exp=1)
                # experiment 2
                if exp2:
                    print("clf_experiment2:" + model_name + '_epoch' + str(num_epochs))
                    all_f1_exp2 = []
                    for i in range(runs):
                        include_real = True
                        fake_data, y_fake, gen = get_fake_rssi(y_train, num_epochs, total_number, include_real,
                                                               low_bound, NUM_CLASSES, APs, save_directory, model_name, device)

                        mix_data = fake_data.view(len(fake_data), APs, 20)
                        # mix_data = torch.transpose(mix_data, 1, 2).cpu().detach().numpy()
                        mix_data = torch.cat((X_train, mix_data), dim=0)

                        # mix_label = y_fake.cpu()
                        mix_label = torch.cat((y_train, y_fake.cpu()), dim=0)

                        f1_exp2 = classification_f1(mix_data, mix_label,X_test, y_test,
                                                    APs, NUM_CLASSES, epochs, train_Bsize, device, False, exp=2,
                                                    flatten=True, show_epoch=False, confusion_met=confusion_matrix)
                        all_f1_exp2.append(f1_exp2)
                        print('exp2 run', i, 'is done')
                    df_gan = generate_dataframe(idx_runs, all_f1_exp2, 'exp2')
                    print_mean_f1(all_f1_exp2, exp=2)
                # experiment 3
                if exp3:
                    print("clf_experiment3: smote")
                    all_f1_exp3 = []
                    for i in range(runs):
                        X_res, y_res = smote_sampling(X_train.numpy(), y_train.numpy(), number_samp=total_number)
                        f1_exp3 = classification_f1(torch.from_numpy(X_res), torch.from_numpy(y_res), X_test, y_test,
                                                    APs, NUM_CLASSES, epochs, train_Bsize, device, False, exp=3,
                                                    flatten=True, show_epoch=False, confusion_met=confusion_matrix)
                        all_f1_exp3.append(f1_exp3)
                        print('exp3 run', i, 'is done')
                    df_smote = generate_dataframe(idx_runs, all_f1_exp3, 'exp3')
                    print_mean_f1(all_f1_exp3, exp=3)

                if exp1 and exp2 and not exp3:
                    df = pd.concat((df_real, df_gan))
                    pd.DataFrame(df).to_csv(
                        "GAN/result_classification/result_csv/"
                        + "clf_noSMOTE" +model_name + '_epoch'+str(num_epochs)+".csv")

                if exp1 and exp2 and exp3:
                    df = pd.concat((df_real, df_gan, df_smote))
                    pd.DataFrame(df).to_csv(
                        "GAN/result_classification/result_csv/"
                        + "clf_" +model_name + '_epoch'+str(num_epochs)+".csv")

                if exp3 and not exp1 and not exp2:
                    df = df_smote
                    pd.DataFrame(df).to_csv(
                        "GAN/result_classification/result_csv/"
                        + "clf_SMOTEonly" + model_name + '_epoch' + str(num_epochs) + ".csv")

            rf = False
            if rf:
                exp1 = True
                exp2 = True
                exp3 = True

                # experiment 1
                if exp1:
                    print("rf_experiment1: no augmentation")
                    all_f1_exp1 = []
                    for i in range(runs):
                        f1_exp1 = rf_classification_f1(X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(), APs,
                                                       model_name, exp=1, flatten=True, feature=False, confusion_met=confusion_matrix)
                        all_f1_exp1.append(f1_exp1)
                        print('exp1 run', i, 'is done')
                    df_real = generate_dataframe(idx_runs, all_f1_exp1, 'exp1')
                    print_mean_f1(all_f1_exp1, exp=1)
                # experiment 2
                if exp2:
                    print("rf_experiment2:" + model_name + '_epoch' + str(num_epochs))
                    all_f1_exp2 = []
                    for i in range(runs):
                        include_real = True
                        fake_data, y_fake, gen = get_fake_rssi(y_train, num_epochs, total_number, include_real,
                                                               low_bound, NUM_CLASSES, APs, save_directory, model_name,
                                                               device)

                        mix_data = fake_data.view(len(fake_data), APs, 20)
                        # mix_data = torch.transpose(mix_data, 1, 2).cpu().detach().numpy()
                        mix_data = torch.cat((X_train, mix_data), dim=0)

                        # mix_label = y_fake.cpu()
                        mix_label = torch.cat((y_train, y_fake.cpu()), dim=0)

                        f1_exp2 = rf_classification_f1(mix_data.numpy(), mix_label.numpy(), X_test.numpy(), y_test.numpy(),
                                                       APs, model_name, exp=2, flatten = True, feature = False, confusion_met = confusion_matrix)
                        all_f1_exp2.append(f1_exp2)
                        print('exp2 run', i, 'is done')
                    df_gan = generate_dataframe(idx_runs, all_f1_exp2, 'exp2')
                    print_mean_f1(all_f1_exp2, exp=2)
                # experiment 3
                if exp3:
                    print("rf_experiment3: smote")
                    all_f1_exp3 = []
                    for i in range(runs):
                        X_res, y_res = smote_sampling(X_train.numpy(), y_train.numpy(), number_samp=total_number)
                        f1_exp3 = rf_classification_f1(X_res, y_res, X_test.numpy(), y_test.numpy(),
                                                       APs, model_name, exp=3, flatten = True, feature = False,
                                                       confusion_met = confusion_matrix)
                        all_f1_exp3.append(f1_exp3)
                        print('exp3 run', i, 'is done')
                    df_smote = generate_dataframe(idx_runs, all_f1_exp3, 'exp3')
                    print_mean_f1(all_f1_exp3, exp=3)

                if exp1 and exp2 and not exp3:
                    df = pd.concat((df_real, df_gan))
                    pd.DataFrame(df).to_csv(
                        "GAN/result_classification/result_csv/"
                        + "rf_noSMOTE" + model_name + '_epoch' + str(num_epochs) + ".csv")

                if exp1 and exp2 and exp3:
                    df = pd.concat((df_real, df_gan, df_smote))
                    pd.DataFrame(df).to_csv(
                        "GAN/result_classification/result_csv/"
                        + "rf_" + model_name + '_epoch' + str(num_epochs) + ".csv")

                if exp3 and not exp1 and not exp2:
                    df = df_smote
                    pd.DataFrame(df).to_csv(
                        "GAN/result_classification/result_csv/"
                        + "rf_SMOTEonly" + model_name + '_epoch' + str(num_epochs) + ".csv")

