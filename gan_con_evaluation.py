import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import torch

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from general_utils import uniq_count
from GAN.gan_utils import load_GAN_model, generate_fake_data, get_fake_rssi,\
    MinMaxScaler, renormalization, windowing, get_mean_stat, get_stats,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_B, map_C, map_D, house_map,\
    binary_convert, num_room_house, train, test, load_house_data, get_col_use
from play_space import feature, augment
from gan_evaluation import multiclass, rf_multiclass
from imblearn.over_sampling import SMOTE



def extract_feature(windowed_data, windowed_label, datatype='test'):
    windowed_data_tp = np.transpose(windowed_data, (0, 2, 1))
    X_train_feature, y_train_feature = feature(windowed_data_tp, windowed_label, datatype='rssi')  # (n,Aps,window)
    if reduce_ap:
        torch.save((X_train_feature, y_train_feature),
                   data_directory + '/gan_data/house_' + house_name + '_train_10hop_reduce_'+str(reduce_ap)+'.pt')
        X_train_feature, y_train_feature = torch.load(
            data_directory + '/gan_data/house_' + house_name + '_train_10hop_reduce_'+str(reduce_ap)+'.pt')
    else:
        torch.save((X_train_feature, y_train_feature), data_directory + '/gan_data/house_'+house_name+'_'+datatype+'_10hop.pt')
    return  X_train_feature, y_train_feature

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    house_all = ['A', 'B', 'C', 'D']
    num_epochs_all = [620, 520, 500, 620]

    for i in range(0,4):
        house_name = house_all[i]
        num_epochs = num_epochs_all[i]  # A 620, B 520, C 440(reduce 160), D 620
        reduce_ap = False
        windowed_data, windowed_label, APs, NUM_CLASSES = \
            load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False)
        windowed_data_fl, windowed_label_fl, APs, NUM_CLASSES =\
            load_house_data(data_directory, house_name, datatype='fl', reduce_ap=False)

        WINDOW_SIZE = 20
        total_number = 1000
        low_bound = None
        GANmodel = "conGAN-CNN_house_" + house_name
        model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
        fake_data, y_fake, gen = get_fake_rssi(windowed_label, num_epochs, total_number, low_bound, NUM_CLASSES, APs, save_directory,
                                               model_name, device)
    # feature FINGERPRINT DATA
    #     X_train_feature, y_train_feature = torch.load(data_directory + '/gan_data/house_'+house_name+'_train_10hop.pt')
    #     y_train_feature = y_train_feature - 1
    # feature FREE LIVING DATA
    #     X_test_feature, y_test_feature = torch.load(data_directory + '/gan_data/house_'+house_name+'_test_10hop.pt')
    #     y_test_feature = y_test_feature - 1
    # extract feature FAKE DATA
    #     fake_data = fake_data.view(len(fake_data), APs, WINDOW_SIZE)
    #     X_fake_feature, y_fake_feature = feature(fake_data.cpu().detach().numpy(), y_fake.cpu().detach().numpy(), datatype='rssi')
    #     torch.save((X_fake_feature, y_fake_feature),data_directory + '/gan_data/house_'+house_name+'_feature_fake_' + str(model_name) + str(num_epochs) + '.pt')
    #     X_fake_feature, y_fake_feature = torch.load(
    #         data_directory + '/gan_data/house_'+house_name+'_feature_fake_' + str(model_name) + str(num_epochs) + '.pt')

        windowed_label = windowed_label - 1
        windowed_label_fl = windowed_label_fl - 1
        epochs = 200

        # SMOTE plot
        X = windowed_data.reshape(windowed_data.shape[0], windowed_data.shape[1] * windowed_data.shape[2])
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, windowed_label)
        X_res = X_res.reshape(X_res.shape[0], X_res.shape[1] // windowed_data.shape[2],
                              X_res.shape[1] // windowed_data.shape[1])

        def plot_all_room(X,y, no_label):
            idx = y.argsort()
            unique_y, counts_y = np.unique(y, return_counts=True)
            counter = []
            accum = 0
            for i in range(len(counts_y)):
                counter.append(accum)
                accum += counts_y[i]
            counter.append(len(X))
            for i in range(0, no_label):
                idx_room = idx[counter[i]:counter[i + 1]]
                idx_room_sort = idx_room[idx_room.argsort()]
                select_idx_room = idx_room_sort[0]

                y_select = y[select_idx_room]+1
                X_select = X[select_idx_room]

                plot_line_rssi_gan(X_select, int(y_select),
                                   house=house_name, house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=True,
                                   reduce=False, save_dir='GAN/smote_visual/house_' + house_name + '/',
                                   plot_idx='house_'+ house_name, model_name='SMOTE', transpose=False)


        plot_all_room(X_res, y_res, NUM_CLASSES)

        # plot_line_rssi_gan(X_res[100], y_res[100],
        #                    house=house_name, house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,
        #                    reduce=False, save_dir='GAN/train_visual/ConGAN_wgp_trans_house_' + house_name + '/',
        #                    plot_idx='SMOTE', model_name='noise_fake_', transpose=False)

        multi = False
        if multi:
            # mlp with RSSI input
            multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                       house_name,house_map[house_name], APs, NUM_CLASSES, GANmodel, device, test_set='flive', exp=1, runs=10, epochs=epochs, show_epoch=False, flatten=True)
            multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                       house_name,house_map[house_name], APs, NUM_CLASSES, GANmodel, device, test_set='flive', exp=2, runs=10, epochs=epochs, show_epoch=False, flatten=True)

            # mlp with features input
            # multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
            #            X_fake_feature, y_fake_feature, APs, NUM_CLASSES,
            #            GANmodel, device, test_set='flive', exp=1, runs=5, epochs=epochs, show_epoch=False, feature=True, flatten=False)
            # multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
            #            X_fake_feature, y_fake_feature, APs, NUM_CLASSES,
            #            GANmodel, device, test_set='flive', exp=2, runs=5, epochs=epochs, show_epoch=False, feature=True, flatten=False)

            smote = True
            if smote:
                from collections import Counter

                X = windowed_data.reshape(windowed_data.shape[0], windowed_data.shape[1] * windowed_data.shape[2])
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X, windowed_label)
                X_res = X_res.reshape(X_res.shape[0], X_res.shape[1] // windowed_data.shape[2],
                                      X_res.shape[1] // windowed_data.shape[1])

                multiclass(X_res, y_res, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                           house_name, house_map[house_name], APs, NUM_CLASSES, GANmodel, device, test_set='flive',
                           exp=3, runs=10, epochs=epochs, show_epoch=False,
                           flatten=True)

        rf = False
        if rf:
            fake_data = fake_data.reshape((-1, APs * WINDOW_SIZE))
            windowed_data = windowed_data.reshape((-1, APs * WINDOW_SIZE))
            windowed_data_fl = windowed_data_fl.reshape((-1, APs * WINDOW_SIZE))

            fake_data = fake_data.cpu().detach().numpy()
            y_fake = y_fake.cpu().detach().numpy()

            rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                          GANmodel, house_name, house_map[house_name], test_set='flive', exp=1, runs=10, feature=False)
            rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                          GANmodel, house_name, house_map[house_name], test_set='flive', exp=2, runs=10, feature=False)
            # smote:
            from imblearn.over_sampling import SMOTE
            # X = windowed_data.reshape(windowed_data.shape[0], windowed_data.shape[1] * windowed_data.shape[2])
            X = windowed_data
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, windowed_label)
            # X_res = X_res.reshape(X_res.shape[0], X_res.shape[1] // windowed_data.shape[2],
            #                       X_res.shape[1] // windowed_data.shape[1])
            #
            # X_res = X_res.reshape((-1, APs * WINDOW_SIZE))

            rf_multiclass(X_res, y_res, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                          GANmodel, house_name, house_map[house_name], test_set='flive', exp=3, runs=10, feature=False)

    rf = False
    if rf:
        # RF with features input
        rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                      X_fake_feature, y_fake_feature, GANmodel, f1_type='macro', test_set='flive', exp=1, runs=5, feature=True)
        rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                      X_fake_feature, y_fake_feature, GANmodel, f1_type='macro', test_set='flive', exp=2, runs=5, feature=True)

        # RF with RSSI input
        fake_data = fake_data.view(len(fake_data), APs, WINDOW_SIZE)
        fake_data = fake_data.cpu().detach().numpy()
        y_fake = y_fake.cpu().detach().numpy()

        fake_data = fake_data.reshape((-1, APs*WINDOW_SIZE))
        windowed_data = windowed_data.reshape((-1, APs*WINDOW_SIZE))
        windowed_data_fl = windowed_data_fl.reshape((-1, APs*WINDOW_SIZE))

        rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                      GANmodel, house_name, house_map, test_set='flive', exp=1, runs=10, feature=False)
        rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                      GANmodel, house_name, house_map, test_set='flive', exp=2, runs=10, feature=False)




    # hand-crafted augmentation
    import csv
    from random import randrange
    from rssi_transform import add_noise, rand_channel_drop


    def mix_aug(X, num_fake, house, augtype=None, noise_range=3.0, noise_mean=0.0):
        augment_X = X
        if house == 'A':
            ap_number = 8
        else:
            ap_number = 11
        for n in range(0, num_fake):
            i = int(np.random.randint(len(X), size=1))

            if augtype == None:
                transform_index = randrange(2)
            elif augtype is not None:
                transform_index = augtype

            if transform_index == 0:  # add noise
                new_X = add_noise(X[i], sigma=noise_range, mean=noise_mean)
            if transform_index == 1:  # randomly select 1 or 2 APs to drop(whole/half)
                new_X = rand_channel_drop(X[i], ch_number_drop=randrange(2) + 1, drop_type=None,
                                          drop_ch_index=None)

            new_X = np.asarray(new_X).reshape(-1, ap_number, 20)
            augment_X = np.vstack([augment_X, new_X])

        return augment_X[len(X):len(augment_X)]


    def hand_aug(X, y, num_fake, house, total_room):
        labels_house = np.full((num_fake[0]), 0)
        fake_house = mix_aug(X[y == 0], num_fake[0], house, noise_range=3.0, noise_mean=0.0)

        for room_number in range(1, total_room):
            labels_room = np.full((num_fake[room_number]), room_number)
            fake_room = mix_aug(X[y == room_number], num_fake[room_number], house, noise_range=3.0, noise_mean=0.0)

            fake_house = np.concatenate((fake_house, fake_room), axis=0)
            labels_house = np.concatenate((labels_house, labels_room), axis=0)

        return fake_house, labels_house


    full_lab_aug = False
    if full_lab_aug:
        model_all = ['rf']
        # total_run = 10
        for model in model_all:
            print('model:', model, 'Full_label_aug ')
            with open('result_csv/new_' + model + 'full_labels_aug.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                # writer.writerows([total_sample])
                all_f1 = []
                total_run = 5
                clf = RandomForestClassifier(random_state=42)
                cv = StratifiedKFold(n_splits=3, shuffle=False)
                param_grid = {
                    'min_samples_leaf': [1, 3, 5, 10],
                    'n_estimators': [30, 50, 100, 200, 300, 500]}

                clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True, n_jobs=-1)
                # generate augmented data
                unique_y, counts_y = np.unique(windowed_label, return_counts=True)
                num_fake = []
                for i in range(0, NUM_CLASSES):
                    if total_number - counts_y[i] > 0:
                        num_fake.append(total_number - counts_y[i])
                    else:
                        num_fake.append(0)
                # generate fake data
                augmentation = True
                if augmentation:
                    rssi_train, y_train = hand_aug(np.transpose(windowed_data, (0, 2, 1)), windowed_label, num_fake, house='C', total_room=9)
                    rssi_train = np.concatenate((rssi_train, np.transpose(windowed_data, (0, 2, 1))), axis=0)
                    y_train = np.concatenate((y_train, windowed_label), axis=None)
                else:
                    rssi_train, y_train = np.transpose(windowed_data, (0, 2, 1)), windowed_label
                # extract features
                rssi_train_feature, y_train_feature = feature(rssi_train, y_train, datatype='rssi')

                clf_grid.fit(rssi_train_feature, y_train_feature)
                print('Best parameters are: {}'.format(clf_grid.best_params_))

                for i in range(0, total_run):
                    if model == 'rf':
                        clf_tune = RandomForestClassifier(
                            min_samples_leaf=clf_grid.best_params_['min_samples_leaf'],
                            n_estimators=clf_grid.best_params_['n_estimators'], )

                    # train the model
                    clf_tune.fit(rssi_train_feature, y_train_feature)
                    rssi_test, y_test = feature(np.transpose(windowed_data_fl, (0, 2, 1)), windowed_label_fl, datatype='rssi')
                    y_pred = clf_tune.predict(rssi_test)
                    f1 = f1_score(y_test, y_pred, average='weighted') * 100
                    all_f1.append((f1))

                    writer.writerow([i + 1, all_f1[i]])

                print('mean F1 = ', sum(all_f1) / len(all_f1))
                print(confusion_matrix(y_test, y_pred))

    # GANmodel = "conGAN-CNN_house_" + house_name
    # save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
    # # generator parameter
    # Z_DIM = 11
    # CHANNELS_RSSI = 1
    # FEATURES_GEN = 32
    # NUM_CLASSES = num_room_house[house_name]
    # APs = len(col_idx_use)
    # WINDOW_SIZE = 20
    # GEN_EMBEDDING = 100
    # # load pretrained generator
    # gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    # gen = load_GAN_model(gen, save_directory, model_name, num_epochs, device)
    # # generate fake data
    # total_number = 1000
    # unique_y, counts_y = np.unique(windowed_label, return_counts=True)
    # # get number of fake data
    # num_fake = []
    # for i in range(0, NUM_CLASSES):
    #     if total_number - counts_y[i] > 0:
    #         num_fake.append(total_number - counts_y[i])
    #     else:
    #         num_fake.append(0)
    # fake_data, y_fake = generate_fake_data(gen, Z_DIM, num_fake=num_fake, total_room=NUM_CLASSES, aps=APs,
    #                                        device=device)