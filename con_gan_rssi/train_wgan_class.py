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
    binary_convert, map_A, map_C, map_D, train, test,\
    num_room_house, house_map, load_house_data

from general_utils import uniq_count
import wandb

def MinMaxScaler(data):
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val

def write_aps(x, t, type, epoch_step):
    writer_ap1.add_scalar("Line Plot"+ type + epoch_step, x[0, :][t], t)
    writer_ap2.add_scalar("Line Plot"+ type + epoch_step, x[1, :][t], t)
    writer_ap3.add_scalar("Line Plot"+ type + epoch_step, x[2, :][t], t)
    writer_ap4.add_scalar("Line Plot"+ type + epoch_step, x[3, :][t], t)
    writer_ap5.add_scalar("Line Plot"+ type + epoch_step, x[4, :][t], t)
    writer_ap6.add_scalar("Line Plot"+ type + epoch_step, x[5, :][t], t)
    writer_ap7.add_scalar("Line Plot"+ type + epoch_step, x[6, :][t], t)
    writer_ap8.add_scalar("Line Plot"+ type + epoch_step, x[7, :][t], t)
    writer_ap9.add_scalar("Line Plot"+ type + epoch_step, x[8, :][t], t)
    writer_ap10.add_scalar("Line Plot"+ type + epoch_step, x[9, :][t], t)
    writer_ap11.add_scalar("Line Plot"+ type + epoch_step, x[10, :][t], t)

def plot_line_rssi_one(rssi_plot, label_plot, label_map=None, ap_map=None, full_scale=False, ymin=-5.0, ymax=5.0):
    rssi_plot = pd.DataFrame(np.transpose(rssi_plot), columns=['g', 'd', 'f', 'k', 'c', 'b', 'a', 'j', 'h', 'i', 'e'])
    if ap_map:
        rssi_plot = rssi_plot.rename(ap_map, axis='columns')
    if label_map:
        label_plot = label_map[label_plot]
    #plot the data
    sns.set_theme(rc={'figure.figsize': (8, 4)})
    sns.lineplot(data=rssi_plot, legend=True)
    # sns.displot(rssi_plot, kind='kde', fill=fill, height=5, aspect=2.5)
    # plt.xlim(-0.1, 21)
    plt.ylim(ymin, ymax) #(-4, 3.5)
    if full_scale:
        plt.ylim(-130, 5)
    # print(label_plot)
    plt.title("label %s" % (label_plot), fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

    # plt.legend(loc='lower left', bbox_to_anchor=(0, 1.01, 1.0, 0.5), markerfirst=True,
    #            mode="expand", borderaxespad=0, ncol=9, handletextpad=0.01, )
    plt.xlabel("Timepoint", fontsize=18)
    # plt.xticks(fontsize=22)
    plt.ylabel("RSSI(dB)", fontsize=18)
    # plt.yticks(fontsize=20)
    plt.tight_layout()

    plt.show()
    plt.close()

if __name__ == "__main__":
    # load dataset
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))

    house_name = 'C'
    reduce_ap = False
    X_train, y_train, APs, NUM_CLASSES = \
        load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False)

    y_train = y_train-1
    X_train = np.transpose(X_train, (0, 2, 1))

    X_train2D = torch.from_numpy(X_train)
    X_train2D = X_train2D.unsqueeze(1)

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).int()

    # Hyperparameters etc.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    WINDOW_SIZE = 20
    CHANNELS_RSSI = 1
    GEN_EMBEDDING = 100
    Z_DIM = 11
    FEATURES_CRITIC = 32
    FEATURES_GEN = 32
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10

    dataset_train = torch.utils.data.TensorDataset(X_train2D.float(), y_train)
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         drop_last=True,)

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(Z_DIM, CHANNELS_RSSI, FEATURES_GEN, NUM_CLASSES, APs, WINDOW_SIZE, GEN_EMBEDDING).to(device)
    critic = Discriminator(CHANNELS_RSSI, FEATURES_CRITIC, NUM_CLASSES, APs, WINDOW_SIZE).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for tensorboard plotting
    fixed_noise = torch.randn(32, Z_DIM, APs, 20).uniform_(0, 1).to(device)
    fixed_noise_plot = fixed_noise[0][0]

    model_name = "ConGAN_wgp_rep_house_" + house_name +"_reduce_" + str(reduce_ap)

    gen.train()
    critic.train()

    NUM_EPOCHS = 1200
    # wandb.init(project=model_name, entity="rssi-augmentation")
    # wandb.config = {
    #     "learning_rate": LEARNING_RATE,
    #     "epochs": NUM_EPOCHS,
    #     "batch_size": BATCH_SIZE,
    #     "window_size": WINDOW_SIZE,
    #     "channels_RSSI": CHANNELS_RSSI,
    #     "gen_embed_size": GEN_EMBEDDING,
    #     "noise_dimention": Z_DIM,
    #     "feature_critic": FEATURES_CRITIC,
    #     "feature_gen": FEATURES_GEN,
    #     "critic_iterations": CRITIC_ITERATIONS,
    #     "Lambda_GP": LAMBDA_GP,
    #     "number_APs": APs,
    #     "number_rooms" : NUM_CLASSES,
    # }

    for epoch in range(NUM_EPOCHS):
    # for epoch in range(500, 1000):
        if epoch == 50:
            pass
        for batch_idx, (real, labels) in enumerate(loader_train):
            real = real.to(device)
            cur_batch_size = real.shape[0]
            labels = labels.to(device)

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, APs, 20).uniform_(0, 1).to(device)
                fake = gen(noise, labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake, labels).reshape(-1)
                gp = gradient_penalty(critic, labels, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # wandb.log({"Loss/disc_train_epoch": loss_critic.item()})
            # wandb.log({"Loss/gen_train_epoch": loss_gen.item()})

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )

        plot_line_rssi_gan(noise[0][0].view(APs, 20).cpu().detach().numpy(), labels[0].cpu().detach().numpy() + 1,
                           house=house_name, house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,
                           reduce=False, save_dir='GAN/train_visual/', model_name='noise_fake_', transpose=True)

        plot_line_rssi_gan(fake[0].view(APs, 20).cpu().detach().numpy(), labels[0].cpu().detach().numpy() + 1,
                           house=house_name, house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,
                           reduce=False, save_dir='GAN/train_visual/', model_name='noise_fake_', transpose=True)

        with torch.no_grad():
            x_fake = fake[0].view(APs, 20)
            x_real = real[0].view(APs, 20)
            cur_labels_numpy = labels.cpu().detach().numpy()
            # print(cur_labels_numpy[0])

            plot_line_rssi_gan(x_fake.cpu().detach().numpy(), cur_labels_numpy[0] + 1, house=house_name,
                               house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,reduce=reduce_ap,
                               save_dir='GAN/train_visual/'+model_name+'/', model_name=str(epoch)+'_fake_', transpose=True)
            plot_line_rssi_gan(x_real.cpu().detach().numpy(), cur_labels_numpy[0] + 1, house=house_name,
                               house_map=house_map[house_name], ymin=-0.1, ymax=1.1, save=False,reduce=reduce_ap,
                               save_dir='GAN/train_visual/'+model_name+'/', model_name=str(epoch)+'_real_', transpose=True)

        if epoch != 0 and epoch % 20 == 0:
            torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(epoch) + '.pt')

    torch.save(gen.state_dict(), save_directory + model_name + '_epoch' + str(NUM_EPOCHS) + '.pt')
    # fake = gen(noise, labels)
    # x_fake = fake[0].view(11, 20)
    # x_real = real[0].view(11, 20)
    # cur_labels_numpy = labels.cpu().detach().numpy()
    # plot_line_rssi_one(x_fake.cpu().detach().numpy(), cur_labels_numpy[0], label_map=label_map, ap_map=ap_map, ymin=-0.1, ymax=1.1)
    # plot_line_rssi_one(x_real.cpu().detach().numpy(), cur_labels_numpy[0], label_map=label_map, ap_map=ap_map, ymin=-0.1, ymax=1.1)

