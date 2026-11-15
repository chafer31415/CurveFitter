import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from numpy import polyfit, polyval
from numpy.linalg import norm
from pprint import pprint


HzPerPLLBit = 2.38  # Global constant

def process_latest_csv(filepath=""):
    if filepath == "":
        script_dir = os.getcwd()
        csv_files = [f for f in os.listdir(script_dir) if f.endswith(".csv")]
        sorted_files = sorted(csv_files, key=lambda f: os.path.getmtime(os.path.join(script_dir, f)), reverse=True)
        pathtoload = os.path.join(script_dir, sorted_files[0])  # Most recently added file
    else:
        pathtoload = filepath

    # Read the first line (metadata)
    with open(pathtoload, 'r') as file:
        reader = csv.reader(file)
        metadata = next(reader)  # First line

    # Read the CSV file while skipping the metadata row and one additional line
    data = np.loadtxt(pathtoload, delimiter=",",skiprows=2, usecols=(0, 1))
    indices = np.arange(len(data[:, 0]))
    timestep = data[1, 0]
    counts = data[:, 1]

    return {"timestep": timestep, "data": (indices, counts)}


def scatter_plot_vector_with_points(trace, points):
    plt.scatter(trace[0], trace[1], label="CPT trace", s=1, color="blue")
    plt.xlabel("Sample")
    plt.ylabel("Counts")
    plt.title("CPT Trace")
    plt.grid(True)

    # Split the given points into x and y coordinates
    x_points, y_points = zip(*points)

    # Overlay the two peak points
    plt.scatter(x_points, y_points, color="red", s=40, label="Peaks", edgecolor="black")

    plt.legend()
    plt.show()


def find_min_max(trace):
    min_count_idx = np.argmin(trace[1])
    min_count = trace[1][min_count_idx]
    max_count_idx = np.argmax(trace[1])
    max_count = trace[1][max_count_idx]

    # Sort the peaks (min and max) in ascending order
    return sorted([[min_count_idx, min_count], [max_count_idx, max_count]])


def slicedata(data, min_idx, max_idx, slicesize=5):
    # Go a little further than the min and max indices
    center = round((max_idx + min_idx) / 2)
    left_lim = max(center - slicesize, 0)
    right_lim = min(center + slicesize, len(data[0]))  # Guard against out-of-range indices

    # Extract the slice of the vector between left_lim and right_lim
    dataslice = (data[0][left_lim:right_lim], data[1][left_lim:right_lim])
    return dataslice


def poly_fit(data, stepsize, order=3):
    # Shift data so that it is centered at (0, 0)
    x = data[0] * stepsize
    xs = x - np.mean(x)
    y = data[1]
    ys = y - np.mean(y)

    # Perform polynomial fitting using `np.polyfit`
    p = np.polyfit(xs, ys, order)
    fitdata = np.polyval(p, xs)

    slope = p[-2]  # The linear term in the polynomial
    residuals = ys - fitdata
    std_residuals = np.std(residuals)

    return slope, std_residuals, [xs, ys], [xs, fitdata]


def sigma(slope, cycletime, stdev):
    measures_in_1_sec = 1.0 / cycletime 
    stderr_counts_1_sec = stdev / np.sqrt(measures_in_1_sec)
    stderr_freq_1_sec = stderr_counts_1_sec / slope
    clock_freq = 6.834E9
    frac_sigma_1_sec = stderr_freq_1_sec / clock_freq

    return abs(frac_sigma_1_sec)


def main(filepath="", slicesize=5, pllstep=10):
    result = process_latest_csv(filepath)
    timestep = result["timestep"]
    cpttrace = result["data"]
    mm = find_min_max(cpttrace)  # Sort so that even if slope is opposite, the points are in the right order
    
    dataslice = slicedata(cpttrace, mm[0][0], mm[1][0],slicesize)
    peaktotroughwidth = abs(mm[0][0]-mm[1][0]) * pllstep * HzPerPLLBit
    slope, stdev, scaleddata, fit = poly_fit(dataslice, pllstep * HzPerPLLBit)
    sigma_value = sigma(slope, timestep, stdev)
    
    # Create a single figure with subplots stacked vertically
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)
    
    # First subplot: Scatter plot with min-max points
    axs[0].scatter(cpttrace[0], cpttrace[1], s=2, color="purple")
    axs[0].scatter([mm[0][0], mm[1][0]], [mm[0][1], mm[1][1]], color="red")
    axs[0].set_title("Scatter Plot with Min-Max Points")
    axs[0].set_xlabel("data point number (arb)")
    axs[0].set_ylabel("cpt signal (counts)")
    axs[0].legend()
    axs[0].annotate(f"peak to trough width in Hz: {peaktotroughwidth:.0f}", 
                    xy=(.3, .8),
                    xycoords="axes fraction"
                    )

    # Second subplot: Scatter plot with polynomial fit
    axs[1].scatter(scaleddata[0], scaleddata[1], s=2, color="blue")
    axs[1].plot(fit[0], fit[1], color="orange")
    axs[1].set_title("Polynomial Fit w/ slope and inferred sigma@1s")
    axs[1].set_xlabel("Hz")
    axs[1].set_ylabel("cpt signal (counts)")
    axs[1].legend()
    axs[1].annotate(f"Slope: {slope:.4f}, Sigma: {sigma_value:.2e}", 
                    xy=(.3,.8),
                    xycoords="axes fraction"
                    )
    plt.show()
    #return slope, stdev, pllstep, sigma_value

main("./cpttraces/highsnrcpt_04_27_2024.csv", 10, 10) #filename, slicesize is +/- away from the middle point, pllstep from GUI