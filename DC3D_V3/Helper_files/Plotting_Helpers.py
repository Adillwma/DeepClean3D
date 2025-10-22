
from Helper_files.Helper_Functions import plot_save_choice
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import torch

# Source: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
def radar_factory(num_vars, frame='circle'):
    """
    Creates a radar chart with `num_vars` Axes.
    This function creates a RadarAxes projection and registers it.
    Source: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes. 
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def create_radar_plot(categories, dictionarys_list, model_names_list):
    """
    Creates a radar plot of the given data

    Args:
        categories (list): A list of the categories to be plotted
        dictionarys_list (list): A list of dictionaries containing the data to be plotted
        model_names_list (list): A list of the names of the models to be plotted

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the radar plot
    
    """
    theta = radar_factory(len(categories), frame='polygon')
    normalisation_factors = [(0, 100), (-2000, 2000), (-2000, 2000), (-1, 1), (-1,1), (0,1), (0,100), (0,100), (200, 0)]

    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    #print("Categories:", categories)    

    # create list of spread out colours from a cmap of len len(dictionarys_list) 
    cmap = plt.get_cmap('viridis')
    colors_list = [cmap(i) for i in np.linspace(0, 1, len(dictionarys_list))]
    for dictionary, color, title in zip(dictionarys_list, colors_list, model_names_list):        
        data_list = []
        for idx, metric in enumerate(categories):
            data_list.append((dictionary[metric][-1] - normalisation_factors[idx][0]) / (normalisation_factors[idx][1] - normalisation_factors[idx][0]))
            #print(metric, " NORMALISED:", (dictionary[metric][-1] - normalisation_factors[idx][0]) / (normalisation_factors[idx][1] - normalisation_factors[idx][0]))

        ax.plot(theta, data_list, color=color, label=title + f"\nAreaScore: {polygon_area_polar(torch.tensor(data_list), torch.tensor(theta))}")
        ax.fill(theta, data_list, facecolor=color, alpha=0.25, label='_nolegend_')
    
            # add title 
        #ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')
    titles = ['MSE', 'SNR', 'PSNR', 'SSIM', 'NMI', 'CC', 'S%', 'T%', 'FP']
    ax.set_varlabels(titles)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_rlim(0, 1)

    # add legend relative to top-left plot
    legend = ax.legend(loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    return fig


def create_tri_radar_plot(categories, dictionarys_list, model_names_list):
    """
    Creates a radar plot of the given data for only the three main metrics

    Args:
        categories (list): A list of the categories to be plotted
        dictionarys_list (list): A list of dictionaries containing the data to be plotted
        model_names_list (list): A list of the names of the models to be plotted

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the radar plot for the three main metrics
    
    """
    theta = radar_factory(3, frame='polygon')
    normalisation_factors = [(0,100), (0,100), (200, 0)]

    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    #print("Categories:", categories)    

    # create list of spread out colours from a cmap of len len(dictionarys_list) 
    cmap = plt.get_cmap('viridis')
    colors_list = [cmap(i) for i in np.linspace(0, 1, len(dictionarys_list))]
    for dictionary, color, title in zip(dictionarys_list, colors_list, model_names_list):        
        data_list = []
        idx = 0
        for metric in categories:
            if metric == 'epoch_avg_loss_true_positive_xy' or metric == 'epoch_avg_loss_true_positive_tof' or metric == 'epoch_avg_loss_false_positive_xy':
                data_list.append((dictionary[metric][-1] - normalisation_factors[idx][0]) / (normalisation_factors[idx][1] - normalisation_factors[idx][0]))
                #print(metric, " NORMALISED:", (dictionary[metric][-1] - normalisation_factors[idx][0]) / (normalisation_factors[idx][1] - normalisation_factors[idx][0]))
                idx += 1

        ax.plot(theta, data_list, color=color, label=title + f"\nAreaScore: {polygon_area_polar(torch.tensor(data_list), torch.tensor(theta))}")
        ax.fill(theta, data_list, facecolor=color, alpha=0.25, label='_nolegend_')
    
            # add title 
        #ax.set_title(title + f"\n{}", weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')
    titles = ['S%', 'T%', 'FP']
    ax.set_varlabels(titles)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_rlim(0, 1)

    # add legend relative to top-left plot
    legend = ax.legend(loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    return fig

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_3d_radar_plot(categories, dictionary, model_name):
    """
    Creates a radar plot of the given data

    Args:
        categories (list): A list of the categories to be plotted
        dictionarys_list (list): A list of dictionaries containing the data to be plotted
        model_names_list (list): A list of the names of the models to be plotted

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the 3d radar plot
    
    """
    theta = radar_factory(len(categories), frame='polygon')
    normalisation_factors = [(0,100), (0,100), (200, 0)]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')



    # Offset between each radar plot
    z_offset = 1


    for epoch in range (len(dictionary[categories[0]])):     
        #print("3d Epoch:", epoch)  
        data_list = []
        for idx, metric in enumerate(categories):
            data_list.append((dictionary[metric][epoch] - normalisation_factors[idx][0]) / (normalisation_factors[idx][1] - normalisation_factors[idx][0]))
            #print(metric, " NORMALISED:", (dictionary[metric][-1] - normalisation_factors[idx][0]) / (normalisation_factors[idx][1] - normalisation_factors[idx][0]))

        #ax.plot(theta, data_list, label=model_name + f"\nAreaScore: {polygon_area_polar(torch.tensor(data_list), torch.tensor(theta))}")
        #ax.fill(theta, data_list, alpha=0.25, label='_nolegend_')
    
        xs = np.array(theta)
        ys = np.array(data_list)
        zs = np.array([epoch * z_offset] * (len(categories) + 1))

        xs = np.concatenate([xs, [xs[0]]])  # Complete the loop
        ys = np.concatenate([ys, [ys[0]]])  # Complete the loop

        print("xs:", xs)
        print("ys:", ys)
        print("zs:", zs)

        # Plot the outline of the radar plot
        ax.plot(xs, zs, ys, label=model_name + f"\nAreaScore: {polygon_area_polar(torch.tensor(ys), torch.tensor(xs))}")
        
        # Create the filled polygon
        verts = [(x, z, y) for x, z, y in zip(xs, zs, ys)]
        poly = Poly3DCollection([verts], alpha=0.25)
        ax.add_collection3d(poly)

    # add legend relative to top-left plot
    ax.legend(loc=(0.9, .95), labelspacing=0.1, fontsize='small')


    # Set the labels and limits
    ax.set_xlabel('Metric')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('Values')
    ax.set_yticks([epoch * z_offset for epoch in range(len(dictionary[categories[0]]))])
    ax.set_yticklabels([f'Slice {epoch+1}' for epoch in range(len(dictionary[categories[0]]))])

    # Set the view angle
    ax.view_init(30, 140)

    return fig

def polygon_area_polar(r, theta):
    """
    Calculate the area of a polygon given its vertices in polar coordinates.
    
    Parameters:
    r (torch.Tensor): Tensor containing the radial distances of the vertices.
    theta (torch.Tensor): Tensor containing the angles of the vertices in radians.
    
    Returns:
    area (float): The area of the polygon.
    """
    # Convert polar coordinates to Cartesian coordinates
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    
    # Shift vertices by one position
    x_shifted = torch.roll(x, shifts=-1)
    y_shifted = torch.roll(y, shifts=-1)
    
    # Calculate the terms in the Shoelace formula
    term1 = torch.dot(x, y_shifted)
    term2 = torch.dot(y, x_shifted)
    
    # Apply the Shoelace formula
    area = 0.5 * torch.abs(term1 - term2)
    
    return area.item()


def loss_plot_trainval(x_list, y_list, train_val_list, legend_label_list, x_label, y_label, title, save_path, plot_or_save):
    """
    Creates a comparative line plot of the given sets of data and saves it to disk

    Args:
        x_list (list): A list of the x axis data
        y_list (list): A list of the y axis data
        train_val_list (list): A list of the training/validation flags
        legend_label_list (list): A list of the labels for the legend
        x_label (str): The label for the x axis
        y_label (str): The label for the y axis
        title (str): The title of the plot
        save_path (str): The path to save the plot to
        plot_or_save (int): A flag to set if the plot is printed to terminal or saved to disk. 0 prints plots to terminal (blocking till closed), If set to 1 then saves all end of epoch printouts to disk (non-blocking), if set to 2 then saves outputs whilst also printing for user (blocking till closed)

    Generates:
        A comparative line plot saved to disk
    """

    for x, y, legend_label, train_val in zip(x_list, y_list, legend_label_list, train_val_list):
        if train_val == 'train':
            plt.plot(x, y, label=legend_label + ' Training Loss')
        else:
            plt.plot(x, y, label=legend_label + ' Validation Loss', linestyle='dashed')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.2)
    plt.legend()
    plot_save_choice(plot_or_save, save_path) 

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
def plot_telemetryOLD(telemetry, true_num_of_signal_points, save_path, plot_or_save):
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
    plt.grid(alpha=0.2)
    plt.legend()
    plot_save_choice(plot_or_save, save_path)

def plot_telemetry(telemetry, true_num_of_signal_points, total_photons, save_path, plot_or_save):
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
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # First plot
    ax1.plot(tele[:,0], tele[:,1], color='r', label="Points above threshold")
    ax1.set_title("Reconstruction threshold telemetry per epoch")
    ax1.set_ylabel("Number of Signal Points")
    ax1.grid(alpha=0.2)

    # check if true_num_of_signal_points is a scalar or a tuple
    if isinstance(true_num_of_signal_points, int):
        ax1.axhline(y=true_num_of_signal_points, color='r', linestyle='dashed', label="True number of signal points")
    else:
        ax1.axhline(y=true_num_of_signal_points[0], color='r', linestyle='dashed', label="True number of signal points")
        ax1.axhline(y=true_num_of_signal_points[1], color='r', linestyle='dashed')


    # Second plot
    ax2.plot(tele[:,0], tele[:,2], color='b', label="Points below threshold")
    ax2.set_xlabel("Epoch number")
    ax2.set_ylabel("Number of Blanks")
    ax2.grid(alpha=0.2)

    # check if true_num_of_signal_points is a scalar or a tuple again?? do it in one! or save a var saying it is true or false
    if isinstance(true_num_of_signal_points, int):
        ax2.axhline(y=total_photons-true_num_of_signal_points, color='b', linestyle='dashed', label="True number of blanks")
    else:
        ax2.axhline(y=total_photons-true_num_of_signal_points[0], color='b', linestyle='dashed', label="True number of blanks")
        ax2.axhline(y=total_photons-true_num_of_signal_points[1], color='b', linestyle='dashed')

    # Combine legends
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=2)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
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
    axs[0, 0].set_title("Mean Sq Error")
    axs[0, 0].set_xlabel("Epoch number")
    axs[0, 0].set_ylabel("MSE")
    axs[0, 0].grid(alpha=0.2) 

    axs[0, 1].plot(epochs_range, epoch_avg_loss_snr)
    axs[0, 1].set_title("Signal to Noise Ratio")
    axs[0, 1].set_xlabel("Epoch number")
    axs[0, 1].set_ylabel("SNR")
    axs[0, 1].grid(alpha=0.2) 

    axs[0, 2].plot(epochs_range, epoch_avg_loss_psnr)
    axs[0, 2].set_title("Peak Signal to Noise Ratio")
    axs[0, 2].set_xlabel("Epoch number")
    axs[0, 2].set_ylabel("PSNR")
    axs[0, 2].grid(alpha=0.2) 

    axs[1, 0].plot(epochs_range, epoch_avg_loss_ssim)
    axs[1, 0].set_title("Structural Similarity Index")
    axs[1, 0].set_xlabel("Epoch number")
    axs[1, 0].set_ylabel("SSIM")
    axs[1, 0].grid(alpha=0.2) 

    axs[1, 1].plot(epochs_range, epoch_avg_loss_nmi)
    axs[1, 1].set_title("Normalised Mutual Information")
    axs[1, 1].set_xlabel("Epoch number")
    axs[1, 1].set_ylabel("NMI")
    axs[1, 1].grid(alpha=0.2) 

    axs[1, 2].plot(epochs_range, epoch_avg_loss_cc)
    axs[1, 2].set_title("Coreelation Coefficent")
    axs[1, 2].set_xlabel("Epoch number")
    axs[1, 2].set_ylabel("CC")
    axs[1, 2].grid(alpha=0.2) 

    axs[2, 0].plot(epochs_range, epoch_avg_loss_true_positive_xy)
    axs[2, 0].set_title("Spatial Retention %")
    axs[2, 0].set_xlabel("Epoch number")
    axs[2, 0].set_ylabel("Signal Spatial Retention (%)")
    axs[2, 0].set_ylim(-5 ,105)
    axs[2, 0].grid(alpha=0.2) 

    axs[2, 1].plot(epochs_range, epoch_avg_loss_true_positive_tof)
    axs[2, 1].set_title("Temporal Retention %")
    axs[2, 1].set_xlabel("Epoch number")
    axs[2, 1].set_ylabel("Signal Temporal Retention (%)")
    axs[2, 1].set_ylim(-5 ,105)
    axs[2, 1].grid(alpha=0.2) 

    axs[2, 2].plot(epochs_range, epoch_avg_loss_false_positive_xy)
    axs[2, 2].set_title("Spatial False Positive")
    axs[2, 2].set_xlabel("Epoch number")
    axs[2, 2].set_ylabel("Number of False Positives")
    axs[2, 2].grid(alpha=0.2)

    plt.tight_layout()


    plot_save_choice(plot_or_save, save_path)