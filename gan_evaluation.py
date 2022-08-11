import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from GAN.con_gan_rssi.model_wgan_gp import Discriminator, Generator, initialize_weights
from general_utils import uniq_count
from GAN.gan_utils import load_GAN_model, generate_fake_data, get_fake_rssi,\
    MinMaxScaler, renormalization, windowing, get_mean_stat, get_stats,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_B, map_C, map_D, house_map,\
    binary_convert, num_room_house, train, test
from play_space import feature


def multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
               APs, NUM_CLASSES, GANmodel, device, test_set='flive', exp=1, runs=1, show_epoch=True,
               feature=False, flatten=True, confusion_met=True, rt=False):

    X1, X2, y1, y2 = train_test_split(windowed_data, windowed_label, test_size=0.3, shuffle=True,
                                      stratify=windowed_label, random_state=42)
    X1, y1 = windowed_data, windowed_label

    # TRAIN set
    if exp == 1:  # Experiment 1
        X_train = torch.from_numpy(X1)
        y_train = torch.from_numpy(y1)
    if exp == 2:  # Experiment 2
        if not feature:
            fake_data = fake_data.view(len(fake_data), APs, 20)
            fake_data = torch.transpose(fake_data, 1, 2)
            fake_data = fake_data.cpu().detach().numpy()
            y_fake = y_fake.cpu().detach().numpy()
        X_train = torch.cat((torch.from_numpy(X1).to(device), torch.from_numpy(fake_data).to(device)), dim=0)
        y_train = torch.cat((torch.from_numpy(y1).to(device), torch.from_numpy(y_fake).to(device)), dim=0)

    # TEST set
    if test_set == 'fp':
        X_test = torch.from_numpy(X2)
        y_test = torch.from_numpy(y2)
    if test_set == 'flive':
        X_test = torch.from_numpy(windowed_data_fl)
        y_test = torch.from_numpy(windowed_label_fl)

    if flatten:
        X_train = torch.reshape(X_train, (-1, APs*20))
        X_test = torch.reshape(X_test, (-1, APs*20))

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.float())
    test_dataset = torch.utils.data.TensorDataset(X_test.float(), y_test.float())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True,
                                               drop_last=True, )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=128,
                                              shuffle=False,
                                              drop_last=True, )

    model_type = "mlp"
    print(test_set, 'experiment', exp, model_type, 'model', GANmodel, 'feature',feature)
    F1_all_1 = []
    acc_all_1 = []
    for i in range(runs):
        # if model_type == "log":
        #     model = LogisticRegression(APs*20, NUM_CLASSES).to(device)
        if model_type == "mlp":
            if feature:
                model = MLPClassifier(APs*7, NUM_CLASSES).to(device)
            else:
                model = MLPClassifier(APs*20, NUM_CLASSES).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        epochs = 200
        for epoch in range(epochs):
            loss_epoch, accuracy_epoch, f1_epoch = train(train_loader, model, optimizer, criterion)
            # train(train_loader, model, optimizer, criterion)
            if show_epoch:
                print(
                    f"Epoch [{epoch}/{epochs}]\t "
                    f"Loss: {loss_epoch / len(train_loader)}\t "
                    f"Accuracy: {accuracy_epoch}\t "
                    f"F1: {f1_epoch * 100}\t")

        loss, accuracy, f1 = test(test_loader, model, criterion)
        F1_all_1.append(f1)
        acc_all_1.append(accuracy)

    print('Average f1: ', sum(F1_all_1) / len(F1_all_1))
    # pred, true = model_test(X_test.float(), y_test, model)
    if confusion_met:
        _output = model(X_test.float())
        pred = _output.argmax(1)
        acc = (pred == y_test).sum().item() / y_test.size(0)
        print('Accuracy:', acc * 100)
        pred = pred.cpu().detach().numpy()
        true = y_test.cpu().detach().numpy()
        f1 = f1_score(true, pred, average='weighted')
        print('F1 score:', f1 * 100)
        print(confusion_matrix(true, pred))
    if rt:
        return sum(acc_all_1)/len(acc_all_1), sum(F1_all_1)/len(F1_all_1)

def rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                  X_fake_feature, y_fake_feature, GANmodel, test_set='fp', exp=1, runs=10, feature=True):

    X1, X2, y1, y2 = train_test_split(X_train_feature, y_train_feature, test_size=0.3, shuffle=True,
                                      stratify=y_train_feature, random_state=42)
    X1, y1 = X_train_feature, y_train_feature

    if test_set == 'fp':
        X_train, X_test, y_train, y_test = X1, X2, y1, y2
    if test_set == 'flive':
        X_train, y_train = X1, y1
        X_test, y_test = X_test_feature, y_test_feature

    if exp == 2:
        X_train = np.concatenate((X_train, X_fake_feature), axis=0)
        y_train = np.concatenate((y_train, y_fake_feature), axis=0)

    print(test_set, 'experiment', exp,'model', GANmodel,'feature',feature)

    clf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=False)
    param_grid = {
        'min_samples_leaf': [1, 3, 5, 10],
        'n_estimators': [30, 50, 100, 200, 300, 500, 1000]}
    clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True, n_jobs=-1)
    clf_grid.fit(X_train_feature, y_train_feature)
    print('Best parameters are: {}'.format(clf_grid.best_params_))

    f1_all = []
    acc_all =[]
    for i in range(runs):
        clf_tune = RandomForestClassifier(min_samples_leaf=clf_grid.best_params_['min_samples_leaf'],
                                          n_estimators=clf_grid.best_params_['n_estimators'], )
        clf_tune.fit(X_train, y_train)
        y_pred = clf_tune.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        acc = accuracy_score(y_test, y_pred)* 100
        f1_all.append(f1)
        acc_all.append(acc)

    print('Average f1:', np.mean(f1_all))
    print('Average acc', np.mean(acc_all))
    print(confusion_matrix(y_test, y_pred))

def get_col_use(house_name, reduce_ap):
    if house_name == 'A':
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        col_idx_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    col_idx_use_label = col_idx_use[len(col_idx_use) - 1] + 1

    if reduce_ap:
        if house_name == 'C':
            col_idx_use = [1, 2, 4, 7, 9, 10]
    return col_idx_use, col_idx_use_label

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # data_directory = os.path.join('..', 'SimCLR', 'localisation', 'data', ''.format(os.path.sep))
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 160
    house_name = 'C'
    reduce_ap = True
    col_idx_use, col_idx_use_label = get_col_use (house_name, reduce_ap)
# load fingerprint data
    house_file = 'csv_house_' + house_name + '_fp.csv'
    ori_data = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[col_idx_use_label])
# load free living data
    house_file = 'csv_house_' + house_name + '_fl.csv'
    ori_data_fl = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
    label_fl = np.loadtxt(data_directory + 'csv_house_C_fl.csv', delimiter=",", skiprows=1, usecols=[col_idx_use_label])
# data normalisation
    norm_data, min_val, max_val = MinMaxScaler(ori_data)
    norm_data_fl, min_val_fl, max_val_fl = MinMaxScaler(ori_data_fl)
# get window data
    windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10)
    windowed_data_fl, windowed_label_fl = windowing(norm_data_fl, label_fl, seq_len=20, hop_size=10)
    check_data = False
    if check_data:
        plot_line_rssi_gan(windowed_data[0], windowed_label[0], transpose=False, house=house_name,
                           house_map=map_D,
                           full_scale=False, ymin=-0.1, ymax=1.1, save=False, model_name='house_D_real')
        get_stats(data_directory, house_file)
    # GENERATE FAKE DATA
    NUM_CLASSES = num_room_house[house_name]
    APs = len(col_idx_use)
    total_number = 1000
    GANmodel = "conGAN-CNN_house_" + house_name
    model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
    fake_data, y_fake, gen = get_fake_rssi(windowed_label, num_epochs, total_number, NUM_CLASSES, APs, save_directory, model_name, device)

    # visualisation
    visualisation = True
    if visualisation:
        for i in range(NUM_CLASSES):
            fake_data_room = fake_data[y_fake == i]
            for j in range(5):
                fake_plot = fake_data_room[j].view(APs, 20)
                plot_line_rssi_gan(fake_plot.cpu().detach().numpy(), i+1, transpose=True, house=house_name, house_map=house_map[house_name],
                                   reduce=reduce_ap, full_scale=False, ymin=-0.1, ymax=1.1, save=True, save_dir='GAN/result_visual/', model_name=model_name+'_e'+str(num_epochs)+'_', plot_idx=j)

    windowed_data_tp = np.transpose(windowed_data, (0, 2, 1))
    X_train_feature, y_train_feature = feature(windowed_data_tp, windowed_label, datatype='rssi') #(n,Aps,window)
    # torch.save((X_train_feature, y_train_feature), data_directory + '/gan_data/house_C_living_train_10hop.pt')
    X_train_feature, y_train_feature = torch.load(data_directory + '/gan_data/house_C_living_train_10hop.pt')
    # windowed_data_fl_tp = np.transpose(windowed_data_fl, (0, 2, 1))
    # X_test_feature, y_test_feature = feature(windowed_data_fl_tp, windowed_label_fl, datatype='rssi')
    # torch.save((X_test_feature, y_test_feature), data_directory + '/gan_data/house_C_living_test_10hop.pt')
    X_test_feature, y_test_feature = torch.load(data_directory + '/gan_data/house_C_living_test_10hop.pt')

    # fake_data_tp = np.transpose(fake_data, (0, 2, 1))
    # fake_data = fake_data.view(len(fake_data), 11, 20)
    # X_fake_feature, y_fake_feature = feature(fake_data.detach().numpy(), y_fake.detach().numpy(), datatype='rssi')
    # torch.save((X_fake_feature, y_fake_feature),data_directory + '/gan_data/house_C_feature_fake_' + str(GANmodel) + str(200) + '.pt')
    X_fake_feature, y_fake_feature = torch.load(
        data_directory + '/gan_data/house_C_feature_fake_' + str(GANmodel) + str(200) + '.pt')

    windowed_label = windowed_label - 1
    windowed_label_fl = windowed_label_fl - 1
    y_train_feature = y_train_feature - 1
    y_test_feature = y_test_feature - 1

    multi = False
    if multi:
        # mlp with RSSI input
        multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                   GANmodel, device, test_set='flive', exp=1, runs=10, show_epoch=False, flatten=True)

        multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                   GANmodel, device, test_set='flive', exp=2, runs=10, show_epoch=False, flatten=True)


        # mlp with features input
        multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                   X_fake_feature, y_fake_feature,
                   GANmodel, device, test_set='flive', exp=1, runs=10, show_epoch=False, feature=True, flatten=False)
        multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                   X_fake_feature, y_fake_feature,
                   GANmodel, device, test_set='flive', exp=2, runs=10, show_epoch=False, feature=True, flatten=False)

    rf = False
    if rf:
        # RF with features input
        rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                  X_fake_feature, y_fake_feature, test_set='flive', exp=1, runs=10,feature=True)
        rf_multiclass(X_train_feature, y_train_feature, X_test_feature, y_test_feature,
                      X_fake_feature, y_fake_feature, test_set='flive', exp=2, runs=10,feature=True)

        # RF with RSSI input
        fake_data = fake_data.view(len(fake_data), 11, 20)
        fake_data = fake_data.cpu().detach().numpy()
        y_fake = y_fake.cpu().detach().numpy()
        fake_data = fake_data.reshape((-1, 220))
        windowed_data =  windowed_data.reshape((-1, 220))
        windowed_data_fl = windowed_data_fl.reshape((-1, 220))

        rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                      test_set='flive', exp=1, runs=10, feature=False)
        rf_multiclass(windowed_data, windowed_label, windowed_data_fl, windowed_label_fl, fake_data, y_fake,
                      test_set='flive', exp=2, runs=10, feature=False)




