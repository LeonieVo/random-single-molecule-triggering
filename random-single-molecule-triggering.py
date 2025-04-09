# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:11:58 2025

@author: Vollmar

"""
import matplotlib.pyplot as plt
import numpy as np
import gillespie  # get from https://github.com/sueskind/gillespie or pip install gillespie
import MarkovFormulas # import helper functions
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import os

if __name__ == '__main__':
    """
 The 3-state model:
       ┌───────┐               ┌───────┐
       │ state │  alpha_for    │  state│
       │   A   │ ────────────► │   B   │
       │OR 0   │ ◄─────────────│OR 1   │
       └──┬────┘  alpha_back   └──┬────┘
          │ ▲                     │ ▲
          │ │                     │ │
gamma_back│ │gamma_for       beta_│ │
          │ │                for  │ │ beta_back
          │ │                     │ │
          │ │     ┌──────┐        │ │
          │ └─────┤state │◄───────┘ │
          └──────►│   C  │──────────┘
                  │OR 2  │
                  └──────┘
                  """
# %% DEFINE YOU MODEL set all parameters:
    # for creating a triggering file in the end:
    exposure_time = 0.200  # in seconds i.e. 200 ms
    readout_time = 0.05  # in seconds i.e. 50 ms
    # THE MODEL:
    N = 1  # whole population, set to 1 for 100%
    # Initial state populations
    initials = [1, 0, 0]  # A, B, C
    # rates for a 3-state model:
    alpha_for = 0.2   # in s^-1, rate for A->B
    beta_for = 0.2    # in s^-1, rate for B->C
    gamma_for = 0.2   # in s^-1, rate for C->A
    alpha_back = 0.1  # in s^-1, rate for A<-B
    beta_back = 0.1   # in s^-1, rate for B<-C
    gamma_back = 0.1  # in s^-1, rate for C<-A
    t = 5  # duration in seconds

    propensities = [lambda a, b, c: alpha_for * a,   # A -> B, Propensity: alpha_forward * A
                    lambda a, b, c: beta_for * b,    # B -> C, Propensity: beta_forward * B
                    lambda a, b, c: gamma_for * c,   # C -> A, Propensity: gamma_forward * C
                    lambda a, b, c: alpha_back * b,  # A <- B, Propensity: alpha_back * B
                    lambda a, b, c: beta_back * c,   # B <- C, Propensity: beta_back * C
                    lambda a, b, c: gamma_back * a]  # C <- A, Propensity: gamma_back * A

    stoichiometry = [[-1, 1, 0],   # A -> B, Population change: A-1, B+1, C+0
                     [0, -1, 1],   # B -> C, Population change: A+0, B-1, C+1
                     [1, 0, -1],   # C -> A, Population change: A+1, B+0, C-1
                     [1, -1, 0],   # A <- B, Population change: A+1, B-1, C+0
                     [0, 1, -1],   # B <- C, Population change: A+0, B+1, C-1
                     [-1, 0, 1]]  # C <- A, Population change: A-1, B+0, C+1
    # %% the gillespie simulation itself
    time_points, ABC = gillespie.simulate(initials, propensities, stoichiometry, t)
    A, B, C = zip(*ABC)
    fig1 = plt.figure()
    # plot1: gillespie results
    plt.plot(time_points, A, label="A")
    plt.plot(time_points, B, label="B")
    plt.plot(time_points, C, label="C")

    plt.title("Gillespie")
    plt.xlabel("seconds")
    plt.ylabel("Population")
    plt.legend(loc='upper right')
    plt.show()
    #fig1.savefig("GillespieSeqnence.pdf", format="pdf", bbox_inches="tight")
    # %% calculate affininty (DeltaG) and entropy production (DeltaS)
    """
    Transitionmatrix:
    The transitionmatrix describes the propapilities of going from one state to 
    another (initial state → final state).
    The transitionmatrix is in terms of propabilities NOT in terms of transition 
    rates.
    rate [s^-1] = prob * framerate [s^-1]
    This is how it looks:
    initial state    
    ↓   final state →  A         │  B                    │ C                    │
    A    │1-alpha_for-gamma_back │alpha_for              │gamma_back            │
    B    │alpha_back             │1-alpha_back-beta_for  │beta_for              │
    C    │gamma_for              │beta_back              │1-gamma_for-beta_back │
    """
    framerate = 1 / (exposure_time + readout_time) # in s^-1
    # Construction of off-diagonal parts of trasion matix
    Transitionmatrix = np.array([[0, alpha_for, gamma_back],
                                [alpha_back, 0, beta_for],
                                [gamma_for, beta_back, 0]
                                ]) / framerate  # convert to probabilities
    # insert diagonal elements: set diagonal as 1 - sum of off-diagonal entries for each row
    np.fill_diagonal(Transitionmatrix, 1 - Transitionmatrix.sum(axis=1))

    # to check if transitionmatrix is okay
    Tcheck = Transitionmatrix / np.sum(Transitionmatrix, axis=1, keepdims=True)
    if not np.array_equal(Tcheck, Transitionmatrix):
        print('There is an error in your Transitionmatrix')
    else:
        print('everything is good with your Transitionmatrix')
    # calculate steady state probabilites
    SteadyStatePi = MarkovFormulas.compute_statPiCalc(Transitionmatrix)  # [probabiliies]
    SteadyStatePop = SteadyStatePi * N  # [particles]
    # calculate DeltaG and DeltaS
    DeltaS = MarkovFormulas.compute_entropy_production(Transitionmatrix, SteadyStatePi)
    print(f"DeltaS = {DeltaS:.2e} kBT")
    DeltaG = MarkovFormulas.compute_cycle_affinity(Transitionmatrix, [0,1,2,0])
    print(f"DeltaG = {DeltaG:.2f} kBT")
    # plot you Hidden Markov model:
    MarkovFormulas.draw_HMM_graph(Transitionmatrix, SteadyStatePi, threshold=1e-3)
    # %% convert Gillespie results into a state sequence
    # Generate equally spaced time vector
    time_vector = np.arange(0, t, exposure_time+readout_time)

    # Convert list into array
    ABC_array = np.array(ABC)
    # Mapping tuples to state indices
    state_mapping = {(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2}
    # Convert each row to a state index
    state_sequence = [state_mapping[tuple(row)] for row in ABC_array]
    # Create state array
    state_array = np.zeros_like(time_vector, dtype=int)
    A_array = np.zeros_like(time_vector, dtype=int)
    B_array = np.zeros_like(time_vector, dtype=int)
    C_array = np.zeros_like(time_vector, dtype=int)
    # Assign states based on time intervals
    for i in range(len(time_points) - 1):
        mask = (time_vector >= time_points[i]) & (time_vector < time_points[i + 1])
        state_array[mask] = state_sequence[i]  # in which state 0 or 1 or 2
        A_array[mask] = A[i] # TRUE (=1) if in state A
        B_array[mask] = B[i] # TRUE (=1) if in state B
        C_array[mask] = C[i] # TRUE (=1) if in state C
    # Assign final state until end of time vector
    state_array[time_vector >= time_points[-1]] = state_sequence[-1]
    A_array[time_vector >= time_points[-1]] = A[-1]
    B_array[time_vector >= time_points[-1]] = B[-1]
    C_array[time_vector >= time_points[-1]] = C[-1]
    # Combine time and state into a single array
    result_array = np.column_stack((time_vector, state_array, A_array, B_array, C_array))
    # %% plots
    # plot state sequence
    fig2 = plt.figure()
    plt.plot(result_array[:, 0], result_array[:, 1])
    plt.title("state sequence")
    plt.xlabel("seconds")
    plt.ylabel("state")
    plt.show()

    # plot when in state A i.e. state 0
    # state A
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(result_array[:, 0], result_array[:, 2], color='green')
    ax1.set_title("state A", fontsize=8)
    ax1.set_ylabel("on/off")
    # state B
    ax2.plot(result_array[:, 0], result_array[:, 3], color='orange')
    ax2.set_title("state B", fontsize=8)
    ax2.set_ylabel("on/off")
    # state C
    ax3.plot(result_array[:, 0], result_array[:, 4], color='red')
    ax3.set_title("state C", fontsize=8)
    ax3.set_ylabel("on/off")
    plt.xlabel("seconds")
    plt.show()
    # %% convert to triggering file
    # create frame time for easier calculation
    frame_time = exposure_time + readout_time  # in seconds
    # create trigger_pointsNEW dict
    trigger_pointsNEW = {}  # in ms
    # define some presents:
    alwaysOff = [(int(t*1000), int(t*1000))]  # in m, default off
    cam0 = (readout_time, readout_time + exposure_time)  # in s, 1 frame switch on camera
    shutterON = (0, frame_time)
    block_time_ms = int(t*1000)  # length of whole triggering block in ms
    # initialize lists for all trigger channels
    cam = [] 
    camBG =[]       
    laserA = []  # green
    laserB = []  # orange
    laserC = []  # red
    shutterBlue = []
    shutterOrange = []
    for frame in range(0, int(t/frame_time), 1):
        # camera:
        cam.append(tuple(int(x*1000+(frame*frame_time*1000)) for x in cam0))  # in ms
        camBG=cam
        if A_array[frame]:
            # laser green on if in state A:
            laserA.append(tuple(int(x*1000+(frame*frame_time*1000)) for x in cam0))
            # if green laser on open Shutter orange:
            shutterOrange.append(tuple(int(x*1000+(frame*frame_time*1000)) for x in shutterON))
        if B_array[frame]:
            # laser orange on if in state B:
            laserB.append(tuple(int(x*1000+(frame*frame_time*1000)) for x in cam0))
            # if orange laser on open Shutter orange:
            shutterOrange.append(tuple(int(x*1000+(frame*frame_time*1000)) for x in shutterON))
        if C_array[frame]:
            # laser red on if in state C:
            laserC.append(tuple(int(x*1000+(frame*frame_time*1000)) for x in cam0))
    # sort entries of shutterOrange to be in ascending order:
    shutterOrange.sort()
    # if orange laser on switch of blue/green cam
    camBG = [value for value, mask in zip(camBG, B_array) if mask == 0]
    # write trigger file:
    trigger_pointsNEW["Cam o/r"] = cam
    trigger_pointsNEW["Cam b/g"] = camBG
    trigger_pointsNEW["Laser blue"] = alwaysOff
    trigger_pointsNEW["Laser green"] = laserA
    trigger_pointsNEW["Laser orange"] = laserB
    trigger_pointsNEW["Laser red"] = laserC
    trigger_pointsNEW["shutter in blue detection"] = alwaysOff
    trigger_pointsNEW["shutter in orange detection"] = shutterOrange
# %% plot trigger file like Anushka
    # Setting the figure size and resolution
    fig = plt.figure(figsize=(9, 4), dpi=300)
    frame_time_ms = int(frame_time*1000)
    framenumber=int(t/frame_time) # both in s
    ax = fig.add_subplot(111)

    y_positions = {
        "Cam o/r": 8.7,
        "Cam b/g": 7.6,
        "Laser blue": 6.5,
        "Laser green": 5.4,
        "Laser orange": 4.3,
        "Laser red": 3.2,
        "shutter in blue detection": 2.1,
        "shutter in orange detection": 1
    }
    colors = {
    "Cam o/r": "black",
    "Cam b/g": "black",
    "Laser blue": "deepskyblue",
    "Laser green": "limegreen",
    "Laser orange": "darkorange",
    "Laser red": "red",
    "shutter in blue detection": "navy",
    "shutter in orange detection": "orange"
    }
    #time_ax = [i / 1000.0 for i in range(0, framenumber+1, 1)]  # 0 to 0.2 sec with 0.001 sec intervals
    time_ax = np.arange(0, t, 1/1000)  # ms time axis

    for name, intervals in trigger_pointsNEW.items():
        signal = [0] * (t*1000) # Initialize all to 0 (off)
        for start, end in intervals:
            for i in range(start, end):  # Mark the signal as on (1) for the specified intervals
                signal[i] = 1

        # Shift the signal up by the y_position of the device
        signal = [s + y_positions[name] for s in signal]

        ax.step(time_ax[:], signal[:], label=name, where='post',color=colors.get(name, 'black'))
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("On/Off States")
    ax.set_title("Trigger Points for Lasers")
    ax.grid(True)
    # ax.legend(loc='upper right')
    plt.show()

# %% write configuration file as txt and save
def write_config_file(save_path, trigger_points, block_time_ms, initials):
    """
    Writes trigger points and block_zeit_ms to a configuration file.
    
    Args:
        save_path (str): Base path for where to save the configuration file.
        trigger_points (dict): Dictionary with device names as keys and lists of (start, end) tuples as values.
        block_time_ms (int): time of whole triggering-block in milliseconds
        initials (list): List containing initial state populations [A, B, C].
    """
    
    try:
        with open(save_path, 'w') as file:
            # Write timestamp and initial state populations
            file.write(f"#{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            file.write(f"# initial state populations: A={initials[0]}, B={initials[1]}, C={initials[2]}\n\n")
            
            # Write block time
            file.write(f'"Block Zeit (ms)":{block_time_ms}\n\n')
            
            # Write trigger points
            for i, (device_name, intervals) in enumerate(trigger_points.items()):
                interval_str = "-".join(f"{start}-{end}" for start, end in intervals)
                file.write(f'channel{i}:"{device_name}":{interval_str}\n')
    
        print(f"Configuration file successfully written to {save_path}")
    except Exception as e:
        print(f"Error writing configuration file: {e}")
# %% Save everything
# 0. Where to save
save_path = "C:\\Users\\Vollmar\\Desktop\\exportTest"
name = "1Test"
figures_to_save = [fig1, fig2]

# 1. Create dated directory
date_str = datetime.now().strftime("%Y%m%d")
dir_name = f"{date_str}_{name}"
full_save_path = os.path.join(save_path, dir_name)
os.makedirs(full_save_path, exist_ok=True)
    
print(f"Saving files to: {full_save_path}")

# 2. Save all figures as individual PNGs and combined PDF
pdf_path = os.path.join(full_save_path, f"{name}_figures.pdf")
with PdfPages(pdf_path) as pdf:
    for i, fig in enumerate(figures_to_save):
        png_path = os.path.join(full_save_path, f"{name}_fig_{i+1}.png")
        fig.savefig(png_path)
        pdf.savefig(fig)
print("Figures saved.")

# 3. Save config file
config_save_path = os.path.join(full_save_path, f"{name}_Laserconfig.txt")
write_config_file(config_save_path, trigger_pointsNEW, block_time_ms, initials)


# 4. Save extra input parameters to another txt
params_path = os.path.join(full_save_path, f"{name}_params.txt")
with open(params_path, "w") as f:
    f.write("Input Parameters:\n")
    f.write(f"population: {N} \n")
    f.write(f"Initial populations: {initials}\n")
    f.write(f"Transition Matrix:\n{Transitionmatrix}\n")
    f.write(f"Exposure Time: {exposure_time} ms\n")
    f.write(f"Readout Time: {readout_time} ms\n")
    
print("Parameters file saved.")