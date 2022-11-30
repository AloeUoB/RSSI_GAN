import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from utils import gradient_penalty
from model_wgan_gp import Discriminator, Generator, initialize_weights

from GAN.gan_utils import load_GAN_model, generate_fake_data,\
    MinMaxScaler, renormalization, windowing, get_mean_stat,\
    plot_line_rssi_gan, MLPClassifier,\
    binary_convert, map_A, map_C, map_D, train, train_val, test,\
    num_room_house, house_map, load_house_data, select_data_potion

from train_wgan_task import select_data
from general_utils import uniq_count
import wandb
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from GAN.gan_evaluation import multiclass, rf_multiclass
from tqdm import tqdm

import copy


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False

def classification_es(X_train, y_train, X_val, y_val, X_test, y_test, APs, NUM_CLASSES, epochs, device, GANmodel,
                   exp=1, runs=1, flatten=True,show_epoch=True, confusion_met=True):
    if flatten:
        X_train = torch.reshape(X_train, (-1, APs * 20))
        X_test = torch.reshape(X_test, (-1, APs * 20))
        X_val = torch.reshape(X_val, (-1, APs * 20))

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    y_val = y_val.long()
    X_val, y_val = X_val.to(device).float(), y_val.to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.long())
    test_dataset = torch.utils.data.TensorDataset(X_test.float(), y_test.long())
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=128,
                                               shuffle=True,drop_last=True, )
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=128,
                                              shuffle=False,drop_last=True, )
    model_type = "mlp"
    print('experiment', exp, model_type, 'GAN-model', GANmodel)
    F1_all_1 = []
    F1_all_2 = []
    mcc_all = []
    acc_all_1 = []
    for i in range(runs):
        model = MLPClassifier(APs * 20, NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        es = EarlyStopping()
        for epoch in range(epochs):
            epoch +=1
            loss_epoch, accuracy_epoch, f1_epoch_w, f1_epoch_m = train(train_loader, model, optimizer, criterion)
            model.eval()
            pred = model(X_val)
            vloss = criterion(pred, y_val)
            if es(model, vloss): done = True
            # pbar.set_description(f"Epoch: {epoch}, tloss: {loss_epoch}, vloss: {vloss:>7f}, EStop:[{es.status}]")
            # else:
            #     pbar.set_description(f"Epoch: {epoch}, tloss {loss_epoch:}")
            print(
                f"Epoch [{epoch}/{epochs}]\t "
                f"tLoss: {loss_epoch / len(train_loader)}\t "
                f"vLoss: {vloss:>7f}\t "
                f"EStop:{es.status}\t"
                f"Accuracy: {accuracy_epoch}\t "
                f"F1_weighted: {f1_epoch_w * 100}\t"
                f"F1_macro: {f1_epoch_m * 100}\t")
        print(epoch)
        loss, accuracy, f1_weight, f1_mac, mcc = test(test_loader, model, criterion)
        F1_all_1.append(f1_weight)
        F1_all_2.append(f1_mac)
        mcc_all.append(mcc)
        acc_all_1.append(accuracy)
        print('Average f1_weighted: ', sum(F1_all_1) / len(F1_all_1))
        print('Average f1_macro: ', sum(F1_all_2) / len(F1_all_2))
        print('Average mcc: ', sum(mcc_all) / len(mcc_all))
        print('Average accuracy: ', sum(acc_all_1) / len(acc_all_1))

        if confusion_met:
            print('This confusion matrix scores')
            _output = model(X_test.float())
            pred = _output.argmax(1)
            acc = (pred == y_test).sum().item() / y_test.size(0)
            print('Accuracy:', acc * 100)
            pred = pred.cpu().detach().numpy()
            true = y_test.cpu().detach().numpy()
            f1_w = f1_score(true, pred, average='weighted')
            f1_m = f1_score(true, pred, average='macro')
            print('F1_weighted score:', f1_w * 100)
            print('F1_macro score:', f1_m * 100)
            print(confusion_matrix(true, pred))

        return loss

def classification_loss(X_train, y_train, X_test, y_test, APs, NUM_CLASSES, epochs, device, GANmodel,
                   exp=1, runs=1, flatten=True, show_epoch=True, confusion_met=True):
    if flatten:
        X_train = torch.reshape(X_train, (-1, APs * 20))
        X_test = torch.reshape(X_test, (-1, APs * 20))

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.long())
    test_dataset = torch.utils.data.TensorDataset(X_test.float(), y_test.long())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               shuffle=True, drop_last=True,)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                              shuffle=False, drop_last=True,)
    model_type = "mlp"
    print('experiment', exp, model_type, 'GAN-model', GANmodel)
    F1_all_1 = []
    F1_all_2 = []
    mcc_all = []
    acc_all_1 = []
    loss_all = []
    for i in range(runs):
        model = MLPClassifier(APs * 20, NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        loss_threshold = 0.0001
        keep_loss = []
        keep_avg_loss = 0
        for epoch in range(epochs):
            epoch +=1
            loss_epoch, accuracy_epoch, f1_epoch_w, f1_epoch_m = train(train_loader, model, optimizer, criterion)
            keep_loss.append(loss_epoch)
            if epoch % 10 == 0 and epoch != 0:

                if abs(keep_avg_loss - (sum(keep_loss)/10)) < loss_threshold:
                    print(
                        f"Epoch [{epoch}/{epochs}]\t "
                        f"Loss: {loss_epoch / len(train_loader)}\t "
                        f"Accuracy: {accuracy_epoch}\t "
                        f"F1_weighted: {f1_epoch_w * 100}\t"
                        f"F1_macro: {f1_epoch_m * 100}\t")
                    break
                else:
                    keep_avg_loss = sum(keep_loss)/10
                keep_loss = []

            if show_epoch:
                print(
                    f"Epoch [{epoch}/{epochs}]\t "
                    f"tLoss: {loss_epoch / len(train_loader)}\t "
                    f"Accuracy: {accuracy_epoch}\t "
                    f"F1_weighted: {f1_epoch_w * 100}\t"
                    f"F1_macro: {f1_epoch_m * 100}\t")

        print('final epoch',epoch)
        loss, accuracy, f1_weight, f1_mac, mcc = test(test_loader, model, criterion)
        F1_all_1.append(f1_weight)
        F1_all_2.append(f1_mac)
        mcc_all.append(mcc*100)
        acc_all_1.append(accuracy)
        loss_all.append(loss)
        print('Average f1_weighted: ', sum(F1_all_1) / len(F1_all_1))
        print('Average f1_macro: ', sum(F1_all_2) / len(F1_all_2))
        print('Average mcc: ', sum(mcc_all) / len(mcc_all))
        print('Average accuracy: ', sum(acc_all_1) / len(acc_all_1))

        if confusion_met:
            print('This confusion matrix scores')
            _output = model(X_test.float())
            pred = _output.argmax(1)
            acc = (pred == y_test).sum().item() / y_test.size(0)
            print('Accuracy:', acc * 100)
            pred = pred.cpu().detach().numpy()
            true = y_test.cpu().detach().numpy()
            f1_w = f1_score(true, pred, average='weighted')
            f1_m = f1_score(true, pred, average='macro')
            print('F1_weighted score:', f1_w * 100)
            print('F1_macro score:', f1_m * 100)
            print(confusion_matrix(true, pred))

        return loss_all

def confuse_acc(X_test, y_test, model):
    output = model(X_test)
    pred = output.argmax(1)
    pred = pred.cpu().detach().numpy()
    true = y_test.cpu().detach().numpy()
    cf_mat = confusion_matrix(true, pred)
    class_acc = []
    for i in range (len(np.unique(true))):
        class_acc.append(cf_mat[i][i]/np.sum(cf_mat[i])*100)

    return cf_mat, class_acc


def plot_class_acc(class_acc, house_map, plot_title):

    label_plot = np.arange(1,len(class_acc))
    label_plot = house_map[label_plot]
    plt.bar(label_plot, class_acc)
    plt.title("%s" % (plot_title), fontsize=14)
    plt.ylim(0, 100)
    plt.show()

def flaten(X):
    return torch.reshape(X, (-1, len(X[0])*len(X[0,0])))

def prep_data(X, y):
    X_prep = torch.from_numpy(np.transpose(X, (0, 2, 1)))
    y_prep = torch.from_numpy(y - 1).int()
    return X_prep, y_prep

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load dataset
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    # load GAN training data
    house_name = 'C'
    reduce_ap = False
    rssi_fp, y_fp, APs, NUM_CLASSES = \
        load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False)

    # X_train, X_val, y_train , y_val = train_test_split(rssi_fp, y_fp, test_size=0.25, random_state=42)
    # X_val = torch.from_numpy(np.transpose(X_val, (0, 2, 1))) #(N, ,)
    # y_val = torch.from_numpy(y_val-1).int()

    # data preparation for training
    X_train, y_train = prep_data(rssi_fp, y_fp)
    X_train2D = X_train.unsqueeze(1)
    # load validation set
    rssi_fl, y_fl, APs, NUM_CLASSES = \
        load_house_data(data_directory, house_name, datatype='fl', reduce_ap=False)
    X_test, y_test = prep_data(rssi_fl, y_fl)

    # custom choose first 10% of each class
    X_val, y_val = select_data_potion(rssi_fl, y_fl, 10, NUM_CLASSES)
    X_val, y_val = prep_data(X_val, y_val)

    epochs = 2000
    cf_loss = classification_loss(X_train, y_train, X_val, y_val, APs, NUM_CLASSES, epochs, device, False,
                   exp=1, runs=10, flatten=True, show_epoch=False, confusion_met=True)

    # # Hyperparameters etc.
    # LEARNING_RATE = 1e-4
    # BATCH_SIZE = 64
    # WINDOW_SIZE = 20
    # CHANNELS_RSSI = 1
    # GEN_EMBEDDING = 100
    # Z_DIM = 11
    # FEATURES_CRITIC = 32
    # FEATURES_GEN = 32
    # CRITIC_ITERATIONS = 5
    # LAMBDA_GP = 10
    # model_name = "ConGAN_wgp_lossC_house_" + house_name + "_reduce_" + str(reduce_ap)
    #
    # # create data dataset & loader
    # dataset_train = torch.utils.data.TensorDataset(X_train2D.float(), y_train)
    # loader_train = torch.utils.data.DataLoader(dataset_train,
    #                                            batch_size=BATCH_SIZE,
    #                                            shuffle=True,
    #                                            drop_last=True,)
    # # initialize gen and disc
    # gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    # critic = Discriminator(CHANNELS_RSSI, FEATURES_CRITIC, NUM_CLASSES, APs, WINDOW_SIZE).to(device)
    # initialize_weights(gen)
    # initialize_weights(critic)
    #
    # # initializate optimizer
    # opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    # opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    #
    # gen.train()
    # critic.train()
    # NUM_EPOCHS = 1200
    # # wandb.init(project=model_name, entity="rssi-augmentation")
    # # wandb.config = {
    # #     "learning_rate": LEARNING_RATE,
    # #     "epochs": NUM_EPOCHS,
    # #     "batch_size": BATCH_SIZE,
    # #     "window_size": WINDOW_SIZE,
    # #     "channels_RSSI": CHANNELS_RSSI,
    # #     "gen_embed_size": GEN_EMBEDDING,
    # #     "noise_dimention": Z_DIM,
    # #     "feature_critic": FEATURES_CRITIC,
    # #     "feature_gen": FEATURES_GEN,
    # #     "critic_iterations": CRITIC_ITERATIONS,
    # #     "Lambda_GP": LAMBDA_GP,
    # #     "number_APs": APs,
    # #     "number_rooms" : NUM_CLASSES,}
    #
    # for epoch in range(NUM_EPOCHS):
    #     for batch_idx, (real, labels) in enumerate(loader_train):
    #         real = real.to(device)
    #         cur_batch_size = real.shape[0]
    #         labels = labels.to(device)
    #
    #         # Train Critic: max E[critic(real)] - E[critic(fake)]
    #         # equivalent to minimizing the negative of that
    #         for _ in range(CRITIC_ITERATIONS):
    #             noise = torch.randn(cur_batch_size, Z_DIM, APs, 20).uniform_(0, 1).to(device)
    #             fake = gen(noise, labels)
    #             critic_real = critic(real, labels).reshape(-1)
    #             critic_fake = critic(fake, labels).reshape(-1)
    #             gp = gradient_penalty(critic, labels, real, fake, device=device)
    #             loss_critic = (
    #                 -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
    #             )
    #             critic.zero_grad()
    #             loss_critic.backward(retain_graph=True)
    #             opt_critic.step()
    #
    #         # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
    #         gen_fake = critic(fake, labels).reshape(-1)
    #         loss_gen = -torch.mean(gen_fake)
    #         gen.zero_grad()
    #         loss_gen.backward()
    #         opt_gen.step()
    #
    #     # wandb.log({"Loss/disc_train_epoch": loss_critic.item()})
    #     # wandb.log({"Loss/gen_train_epoch": loss_gen.item()})
    #
    #     print(
    #         f"Epoch [{epoch}/{NUM_EPOCHS}] \
    #                           Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
    #         )
    #
    #     # plot_line_rssi_gan(fake[0].view(APs, 20).cpu().detach().numpy(), labels[0].cpu().detach().numpy() + 1,
    #     #                    house=house_name, house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,
    #     #                    reduce=False, save_dir='GAN/train_visual/', model_name='noise_fake_', transpose=True)
    #
    #     # classification test
    #
    #
    #     with torch.no_grad():
    #         pass
    #
    #
    #     if epoch != 0 and epoch % 20 == 0:
    #         torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(epoch) + '.pt')
    #
    # torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(NUM_EPOCHS) + '.pt')
