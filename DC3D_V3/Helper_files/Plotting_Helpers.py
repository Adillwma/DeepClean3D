
from Helper_files.Helper_Functions import plot_save_choice
import matplotlib.pyplot as plt
import pickle
import numpy as np


def loss_plot(x, y, x_label, y_label, title, save_path, plot_or_save):
    """
    Creates a simple line plot of the given data and saves it to disk

    Args:
        x (list): The x axis data
        y (list): The y axis data
        x_label (str): The label for the x axis
        y_label (str): The label for the y axis
        title (str): The title of the plot
        save_path (str): The path to save the plot to
        plot_or_save (int): A flag to set if the plot is printed to terminal or saved to disk. 0 prints plots to terminal (blocking till closed), If set to 1 then saves all end of epoch printouts to disk (non-blocking), if set to 2 then saves outputs whilst also printing for user (blocking till closed)

    Generates:
        A simple line plot saved to disk
    """

    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.2)
    plot_save_choice(plot_or_save, save_path) 

def comparitive_loss_plot(x_list, y_list, legend_label_list, x_label, y_label, title, save_path, plot_or_save):
    """
    Creates a comparative line plot of the given sets of data and saves it to disk

    Args:
        x_list (list): A list of the x axis data
        y_list (list): A list of the y axis data
        legend_label_list (list): A list of the labels for the legend
        x_label (str): The label for the x axis
        y_label (str): The label for the y axis
        title (str): The title of the plot
        save_path (str): The path to save the plot to
        plot_or_save (int): A flag to set if the plot is printed to terminal or saved to disk. 0 prints plots to terminal (blocking till closed), If set to 1 then saves all end of epoch printouts to disk (non-blocking), if set to 2 then saves outputs whilst also printing for user (blocking till closed)

    Generates:
        A comparative line plot saved to disk
    """


    for x, y, legend_label in zip(x_list, y_list, legend_label_list):
        plt.plot(x, y, label=legend_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.2)
    plt.legend()
    plot_save_choice(plot_or_save, save_path) 

# Plots the confidence telemetry data
def plot_telemetry(telemetry, true_num_of_signal_points, save_path, plot_or_save):
    """
    Plots the 'lit pixel count' and 'unlit pixel count' telemetry data over time, compared to true number of signal points for comparison. Used to track the recovery performance.

    Args:
        telemetry (list): A list of the telemetry data to be plotted
        true_num_of_signal_points (int): The true number of signal points in the data set
        plot_or_save (int): A flag to set if the plot is printed to terminal or saved to disk. 0 prints plots to terminal (blocking till closed), If set to 1 then saves all end of epoch printouts to disk (non-blocking), if set to 2 then saves outputs whilst also printing for user (blocking till closed)

    Generates:
        A plot of the telemetry data, saved to disk or shown depending on program wide 'plot_or_save' setting
    """
    tele = np.array(telemetry)
    plt.plot(tele[:,0],tele[:,1], color='r', label="Points above threshold") #red = num of points above threshold
    plt.plot(tele[:,0],tele[:,2], color='b', label="Points below threshold") #blue = num of points below threshold
    plt.axhline(y=true_num_of_signal_points, color='g', linestyle='dashed', label="True number of signal points")
    plt.title("Telemetry over epochs")
    plt.xlabel("Epoch number")
    plt.ylabel("Number of Signal Points")
    plt.legend()
    plot_save_choice(plot_or_save, save_path)


def load_comparative_data(comparative_loss_paths, plot_live_training_loss, plot_live_time_loss):
    comparative_history_da = []
    comparative_epoch_times = []
    for loss_path in comparative_loss_paths:
        #load pkl file into dictionary
        if plot_live_training_loss or plot_live_time_loss:
            with open(loss_path + '\\Raw_Data_Output\\history_da_dict.pkl', 'rb') as f:
                comparative_history_da.append(pickle.load(f))
        if plot_live_time_loss:
            with open(loss_path + '\\Raw_Data_Output\\epoch_times_list_list.csv', 'rb') as f:
                # load the data from the csv file called f into a list 
                comparative_epoch_times.append(np.loadtxt(f, delimiter=',').tolist())
    return comparative_history_da, comparative_epoch_times       


def draw_detailed_performance_loss_plots(epochs_range, epoch_avg_loss_mse, epoch_avg_loss_snr, epoch_avg_loss_psnr, epoch_avg_loss_ssim, epoch_avg_loss_nmi, epoch_avg_loss_cc, epoch_avg_loss_true_positive_xy, epoch_avg_loss_true_positive_tof, epoch_avg_loss_false_positive_xy, save_path, plot_or_save):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axs[0, 0].plot(epochs_range, epoch_avg_loss_mse)
    axs[0, 0].set_title("MAE loss")
    axs[0, 0].set_xlabel("Epoch number")
    axs[0, 0].set_ylabel("Loss (MAE)")
    axs[0, 0].grid(alpha=0.2) 

    axs[0, 1].plot(epochs_range, epoch_avg_loss_snr)
    axs[0, 1].set_title("SNR loss")
    axs[0, 1].set_xlabel("Epoch number")
    axs[0, 1].set_ylabel("Loss (SNR)")
    axs[0, 1].grid(alpha=0.2) 

    axs[0, 2].plot(epochs_range, epoch_avg_loss_psnr)
    axs[0, 2].set_title("PSNR loss")
    axs[0, 2].set_xlabel("Epoch number")
    axs[0, 2].set_ylabel("Loss (PSNR)")
    axs[0, 2].grid(alpha=0.2) 

    axs[1, 0].plot(epochs_range, epoch_avg_loss_ssim)
    axs[1, 0].set_title("SSIM loss")
    axs[1, 0].set_xlabel("Epoch number")
    axs[1, 0].set_ylabel("Loss (SSIM)")
    axs[1, 0].grid(alpha=0.2) 

    axs[1, 1].plot(epochs_range, epoch_avg_loss_nmi)
    axs[1, 1].set_title("NMI loss")
    axs[1, 1].set_xlabel("Epoch number")
    axs[1, 1].set_ylabel("Loss (NMI)")
    axs[1, 1].grid(alpha=0.2) 

    axs[1, 2].plot(epochs_range, epoch_avg_loss_cc)
    axs[1, 2].set_title("Coreelation Coefficent? loss")
    axs[1, 2].set_xlabel("Epoch number")
    axs[1, 2].set_ylabel("Loss (CC)")
    axs[1, 2].grid(alpha=0.2) 

    axs[2, 0].plot(epochs_range, epoch_avg_loss_true_positive_xy)
    axs[2, 0].set_title("True Positive XY loss")
    axs[2, 0].set_xlabel("Epoch number")
    axs[2, 0].set_ylabel("Loss (True Positive XY %)")
    axs[2, 0].set_ylim(-5 ,105)
    axs[2, 0].grid(alpha=0.2) 

    axs[2, 1].plot(epochs_range, epoch_avg_loss_true_positive_tof)
    axs[2, 1].set_title("True Positive TOF loss")
    axs[2, 1].set_xlabel("Epoch number")
    axs[2, 1].set_ylabel("Loss (True Positive TOF %)")
    axs[2, 1].set_ylim(-5 ,105)
    axs[2, 1].grid(alpha=0.2) 

    axs[2, 2].plot(epochs_range, epoch_avg_loss_false_positive_xy)
    axs[2, 2].set_title("False Positive XY loss BROKEN?")
    axs[2, 2].set_xlabel("Epoch number")
    axs[2, 2].set_ylabel("Loss (False Positive XY)")
    axs[2, 2].grid(alpha=0.2)

    plot_save_choice(plot_or_save, save_path)