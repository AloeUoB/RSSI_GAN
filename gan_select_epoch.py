import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from sklearn.model_selection import train_test_split
from general_utils import uniq_count
from GAN.gan_utils import load_GAN_model, generate_fake_data, get_fake_rssi,\
    MinMaxScaler, renormalization, windowing, get_mean_stat,\
    plot_line_rssi_gan, MLPClassifier, map_A, map_C, map_D,\
    binary_convert, num_room_house, train, test, load_house_data

from gan_classification import smote_sampling, prep_data
from gan_evaluation import classification_f1




def mm_and_index(score_array, score_type):
    print('f1', score_type, '\t')
    min_f1 = min(score_array)
    min_index = score_array.index(min_f1)
    min_epoch = idx_epoch[min_index]
    print('min epoch:', min_epoch, 'min F1:', min_f1)

    max_f1 = max(score_array)
    max_index = score_array.index(max_f1)
    max_epoch = idx_epoch[max_index]
    print('max epoch:', max_epoch, 'max F1:', max_f1)


def generate_dataframe(idx_epoch, all_f1, label):
    array_data = np.transpose(np.concatenate((np.array([idx_epoch]), np.array([all_f1]))))
    df = pd.DataFrame(array_data, columns=['epochs', 'f1_macro'])
    df['train_data'] = label
    return df

if __name__ == "__main__":
    data_directory = os.path.join('..', 'aloe', 'localisation', 'data', ''.format(os.path.sep))
    save_directory = os.path.join('..', 'aloe', 'GAN', 'save_GANs', ''.format(os.path.sep))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.device_count())
    house_all = ['A', 'B', 'C', 'D']
    shot_all = [10, 5, 3, 1]
    run_map = {10:4, 5:4, 3:4, 1:4}
    smote = False
    for shot in shot_all:
        run_number = run_map[shot]
        for house_name in house_all[0:4]:
            if house_name == 'A':
                smote = True
            else:
                smote = False

            reduce_ap = False
            windowed_data, windowed_label, APs, NUM_CLASSES = \
                load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False)
            # windowed_data_fl, windowed_label_fl, APs, NUM_CLASSES =\
            #     load_house_data(data_directory, house_name, datatype='fl', reduce_ap=False)

            X_train, X_test, y_train, y_test = train_test_split(windowed_data, windowed_label,
                                                                test_size = 0.40,
                                                                shuffle=False,
                                                                random_state=42)
            X_train, y_train = prep_data(X_train, y_train)
            X_test, y_test = prep_data(X_test, y_test)

            total_number = 500

            all_f1_real = []
            all_f1_mix = []
            all_f1_fake = []
            all_f1_sm = []
            all_f1_sm_real = []
            idx_epoch = []
            # model_name = "ConGAN_wgp_rep_house_" + house_name + "_reduce_" + str(reduce_ap)
            # model_name = "ConGAN_wgp_Trans_house_" + house_name +'gen'
            model_name = "ConGAN_wgp_Transhot" + str(shot) + "_house_" + house_name + "_run_" + str(run_number)+'gen'

            print(model_name)
            runs = 1
            for num_epochs in range(20, 2020, 20):
                if house_name == 'A':
                    f1_real = classification_f1(X_train, y_train, X_test, y_test,
                                                APs, NUM_CLASSES, 1200, 128, device, False, exp=1,
                                                flatten=True, show_epoch=False, confusion_met=False)
                if house_name == 'B':
                    f1_real = 55
                if house_name == 'C':
                    f1_real = 62
                if house_name == 'D':
                    f1_real = 51

                if smote:
                    X_res, y_res = smote_sampling(X_train.numpy(), y_train.numpy(), number_samp=total_number)
                    X_sm, y_sm = X_res[len(X_train):len(X_res)], y_res[len(X_train):len(X_res)]

                    f1_sm = classification_f1(torch.from_numpy(X_sm), torch.from_numpy(y_sm),
                                                X_test, y_test,
                                                APs, NUM_CLASSES, 1200, 128, device, False, exp=1,
                                                flatten=True, show_epoch=False, confusion_met=False)

                    f1_real_sm = classification_f1(torch.from_numpy(X_res), torch.from_numpy(y_res),
                                                X_test, y_test,
                                                APs, NUM_CLASSES, 1200, 128, device, False, exp=1,
                                                flatten=True, show_epoch=False, confusion_met=False)
                    all_f1_sm.append(f1_sm)
                    all_f1_sm_real.append(f1_real_sm)

                f1_mix_avg = []
                f1_fake_avg = []
                for i in range(runs):
                    include_real = True
                    fake_data, y_fake, gen = get_fake_rssi(y_train, num_epochs, total_number, include_real, None,
                                                           NUM_CLASSES, APs, save_directory, model_name, device)

                    mix_data = fake_data.view(len(fake_data), APs, 20)
                    mix_data = torch.cat((X_train, mix_data), dim=0)
                    mix_label = torch.cat((y_train, y_fake.cpu()), dim=0)

                    f1_mix = classification_f1(mix_data, mix_label, X_test, y_test,
                                             APs, NUM_CLASSES, 1200, 128, device, False, exp=2,
                                             flatten=True, show_epoch=False, confusion_met=False)

                    f1_fake = classification_f1(fake_data, y_fake, X_test, y_test,
                                                APs, NUM_CLASSES, 1200, 128, device, False, exp="fake only",
                                                flatten=True, show_epoch=False, confusion_met=False)

                    f1_mix_avg.append(f1_mix)
                    f1_fake_avg.append(f1_fake)

                f1_mix = sum(f1_mix_avg)/len(f1_mix_avg)
                f1_fake = sum(f1_fake_avg)/len(f1_fake_avg)

                if smote:
                    print(
                        f"Epoch [{num_epochs}/{2000}]\t "
                        f"F1mac(real): {f1_real}\t"
                        f"F1mac(real+fake): {f1_mix}\t"
                        f"F1mac(fake): {f1_fake}\t"
                        f"F1mac(real+smote): {f1_real_sm}\t"
                        f"F1mac(smote): {f1_sm}\t"
                    )
                else:
                    print(
                        f"Epoch [{num_epochs}/{2000}]\t "
                        f"F1mac(real): {f1_real}\t"
                        f"F1mac(real+fake): {f1_mix}\t"
                        f"F1mac(fake): {f1_fake}\t")

                all_f1_real.append(f1_real)
                all_f1_mix.append(f1_mix)
                all_f1_fake.append(f1_fake)
                idx_epoch.append(num_epochs)

            mm_and_index(all_f1_real, 'real train')
            mm_and_index(all_f1_mix, 'fake+real train')
            mm_and_index(all_f1_fake, 'fake train')
            if smote:
                mm_and_index(all_f1_sm, 'smote train')
                mm_and_index(all_f1_sm_real, 'smote+real train')

            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd

            df_real = generate_dataframe(idx_epoch, all_f1_real, 'real')
            df_gan_real = generate_dataframe(idx_epoch, all_f1_mix, 'gan+real')
            df_gan = generate_dataframe(idx_epoch, all_f1_fake, 'gan')
            if smote:
                df_smote_real = generate_dataframe(idx_epoch, all_f1_sm_real, 'smote+real')
                df_smote = generate_dataframe(idx_epoch, all_f1_sm, 'smote')
                df = pd.concat((df_real, df_gan, df_gan_real, df_smote_real, df_smote))
            else:
                df = pd.concat((df_real, df_gan, df_gan_real))

            pd.DataFrame(df).to_csv(
                "GAN/result_classification/result_csv/" + "epochs_" + model_name + ".csv")

            sns.set_theme(rc={'figure.figsize': (8,4)})
            sns.set(style="ticks")
            sns.lineplot(data=df, x="epochs", y="f1_macro", hue="train_data")
            plt.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0)
            plt.grid()
            plt.tight_layout()
            plt.savefig("GAN/result_classification/line_plot_f1_epoch_AllinOne_"+ model_name+'.png')
            # plt.show()

            sns.set_theme(rc={'figure.figsize': (6,8)})
            sns.set(style="ticks")
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(4,1)
            sns.lineplot(ax=axes[0], data=df, x="epochs", y="f1_macro", hue="train_data")
            # sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1))
            # plt.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0)
            axes[0].get_legend().remove()
            axes[0].set_title('F1 macro comparison')
            sns.lineplot(ax=axes[1], data=df_real, x="epochs", y="f1_macro", color='blue')
            axes[1].set_title('F1 macro real')
            sns.lineplot(ax=axes[2], data=df_gan_real, x="epochs", y="f1_macro", color='green')
            axes[2].set_title('F1 macro GAN+real')
            sns.lineplot(ax=axes[3],data=df_gan, x="epochs", y="f1_macro", color='orange')
            axes[3].set_title('F1 macro GAN')
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig("GAN/result_classification/line_plot_f1_epoch_seperate"+ model_name+'.png')
            # plt.show()

