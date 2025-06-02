# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:11:58 2025

@author: Vollmar

"""
import matplotlib.pyplot as plt
import numpy as np
import gillespie  # get from https://github.com/sueskind/gillespie or pip install gillespie
import MarkovFormulas  # import helper functions
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
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
# %% Where to save export
    save_path = "Z:\\_personalDATA\\JS+LV_4F-TIRF\\007_project_3stateDNA\\generatedTriggerPatterns"
    trigger_name = "for1Hzback033Hz_v2"
    # Do you want to cut the triggering file into chunks?
    cut_it = 1 # yes = 1, no = 0
    n_chunks = 5 # set how many chunks here
# %% DEFINE YOU MODEL set all parameters:
    t = 200*5  # duration in seconds, devided into n_chunks if cut_it=1
    # for creating a triggering file in the end:
    exposure_time = 0.200  # in seconds i.e. 200 ms
    readout_time = 0.050 # in seconds i.e. 50 ms
    # THE MODEL:
    N = 1  # whole population, set to 1 for 100%
    # Initial state populations
    initials = [1, 0, 0]  # A, B, C
    # rates for a 3-state model:
    alpha_for = 1   # in s^-1, rate for A->B
    beta_for = 1   # in s^-1, rate for B->C
    gamma_for = 1   # in s^-1, rate for C->A
    alpha_back = 0.33  # in s^-1, rate for A<-B
    beta_back = 0.33   # in s^-1, rate for B<-C
    gamma_back = 0.33  # in s^-1, rate for C<-A
    

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

    # figure 1: gillespie results
    fig1 = plt.figure()
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
    # %% plot Hidden Markov model:
    fig2 = plt.figure()
    MarkovFormulas.draw_HMM_graph(Transitionmatrix, SteadyStatePi, threshold=1e-3)
    print(fig2.axes[0].get_title())
    # %% convert Gillespie results into a state sequence FRAMEWISE

    # Generate equally spaced time vector
    time_vector = np.arange(0, t, exposure_time+readout_time)
    time_vector_mslike = np.arange(0, t, 1/1000)
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
    state_array = [int(s) for s in state_array] # only integers make sense for the states
    A_array[time_vector >= time_points[-1]] = A[-1]
    B_array[time_vector >= time_points[-1]] = B[-1]
    C_array[time_vector >= time_points[-1]] = C[-1]
    # Combine time and state into a single array
    result_array = np.column_stack((time_vector, state_array, A_array, B_array, C_array))
    # %% convert Gillespie results into a state sequence 1000ms-WISE
    time_vector_mslike = np.arange(0, t, 1/1000)
    # Convert list into array
    ABC_array_ms = np.array(ABC)
    # Create state array
    state_array_ms = np.zeros_like(time_vector_mslike, dtype=int)
    A_array_ms = np.zeros_like(time_vector_mslike, dtype=int)
    B_array_ms = np.zeros_like(time_vector_mslike, dtype=int)
    C_array_ms = np.zeros_like(time_vector_mslike, dtype=int)
    # Assign states based on time intervals
    for i in range(len(time_points) - 1):
        mask = (time_vector_mslike >= time_points[i]) & (time_vector_mslike < time_points[i + 1])
        state_array_ms[mask] = state_sequence[i]  # in which state 0 or 1 or 2
        A_array_ms[mask] = A[i] # TRUE (=1) if in state A
        B_array_ms[mask] = B[i] # TRUE (=1) if in state B
        C_array_ms[mask] = C[i] # TRUE (=1) if in state C
    # Assign final state until end of time vector
    state_array_ms[time_vector_mslike >= time_points[-1]] = state_sequence[-1]
    state_array_ms = [int(s) for s in state_array_ms] # only integers make sense for the states
    A_array_ms[time_vector_mslike >= time_points[-1]] = A[-1]
    B_array_ms[time_vector_mslike >= time_points[-1]] = B[-1]
    C_array_ms[time_vector_mslike >= time_points[-1]] = C[-1]
    # Combine time and state into a single array
    result_array_ms = np.column_stack((time_vector_mslike, state_array_ms, A_array_ms, B_array_ms, C_array_ms))
    # %% state sequence plots
    # plot state sequence
    fig3 = plt.figure()
    plt.plot(result_array[:, 0], result_array[:, 1])
    plt.title("state sequence")
    plt.xlabel("seconds")
    plt.ylabel("state")
    plt.show()

    # plot when in state A i.e. state 0
    # state A
    fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
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
    fig4.suptitle('Sequences for each state')
    plt.show()
    # %% Dwell times
    # create frame time for easier calculation
    frame_time = exposure_time + readout_time  # in seconds
    # TRUE Dwell times from Gillespie i.e. "in continous time"
    # 1. Calculate dwell times
    dwell_times_Gill = [time_points[i+1]-time_points[i] for i in range(len(time_points) - 1)]
    # 2. Split dwell times by state
    state_dwell_times_Gill = defaultdict(list)

    for i in range(len(dwell_times_Gill)):
        state = state_sequence[i]
        state_dwell_times_Gill[state].append(dwell_times_Gill[i])
    
    # Plot histogram for each state
    fig5 = plt.figure()  #figsize=(12, 4)
    plt.hist(state_dwell_times_Gill[0], bins=50, color='green', alpha=0.6, label='State A')
    plt.hist(state_dwell_times_Gill[1], bins=50, color='orange', alpha=0.6, label='State B')
    plt.hist(state_dwell_times_Gill[2], bins=50, color='red', alpha=0.6, label='State C')

    plt.xlabel('Dwell Time [s]')
    plt.ylabel('Frequency')
    plt.title('Dwell Time Histograms Gillespie')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # # BIASED BY FRAMES Dwell times from trigger sequence i.e. "in discrete time/ in frames"
    # # 1. Find dwell times
    # state_dwell_times_Laser = defaultdict(list)
    # current_state = state_array[0]
    # start_index = 0
    
    # for i in range(1, len(state_array)):
    #     if state_array[i] != current_state:
    #         start_time = time_vector[start_index]
    #         end_time = time_vector[i - 1] + (time_vector[1] - time_vector[0])  # include duration of last sample
    #         dwell_duration = end_time - start_time
    #         state_dwell_times_Laser[current_state].append(dwell_duration)
    
    #         # Update for next segment
    #         current_state = state_array[i]
    #         start_index = i
    # # 2. Handle the last segment
    # start_time = time_vector[start_index]
    # end_time = time_vector[-1] + (time_vector[1] - time_vector[0])
    # dwell_duration = end_time - start_time
    # state_dwell_times_Laser[current_state].append(dwell_duration)
    # # Plot histogram for each state
    # fig6 = plt.figure()  #figsize=(12, 4)
    # bins_laser = np.arange(0, max(max(state_dwell_times_Laser.values()))+1, frame_time)
    # bins_Gill = np.arange(0, max(max(state_dwell_times_Laser.values()))+1, frame_time/10)
    # # dwell times from LASER
    # plt.hist(state_dwell_times_Laser[0], bins=bins_laser, color='green', alpha=0.6, label='State A Laser')
    # plt.hist(state_dwell_times_Laser[1], bins=bins_laser, color='orange', alpha=0.6, label='State B Laser')
    # plt.hist(state_dwell_times_Laser[2], bins=bins_laser, color='red', alpha=0.6, label='State C Laser')
    # # dwell times from Gillespie
    # plt.hist(state_dwell_times_Gill[0], bins=bins_Gill, color='darkgreen', alpha=0.6, label='State A Gill')
    # plt.hist(state_dwell_times_Gill[1], bins=bins_Gill, color='darkorange', alpha=0.6, label='State B Gill')
    # plt.hist(state_dwell_times_Gill[2], bins=bins_Gill, color='maroon', alpha=0.6, label='State C Gill')
    # plt.xlabel('Dwell Time [s]')
    # plt.ylabel('Frequency')
    # plt.title('Dwell Time Histograms LASER')
    # plt.xlim((0, max(max(state_dwell_times_Laser.values()))+1))
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # %% convert to triggering file
    # create frame time for easier calculation
    frame_time = exposure_time + readout_time  # in seconds
    # create trigger_pointsNEW dict
    trigger_pointsNEW = {}  # in ms
    # define some presents:
    alwaysOff = [(int(t*1000), int(t*1000))]  # in ms, default off
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
    # cameras in frames:
    for frame in range(0, int(t/frame_time), 1):
        # camera:
        cam.append(tuple(int(x*1000+(frame*frame_time*1000)) for x in cam0))  # in ms
        camBG=cam
    time_points_ms = [round(tp * 1000) for tp in time_points]
    if time_points_ms[-1] > t*1000:  # delete last time_point if it is  bigger than simulation duration becasue this causes problems in triggering
        time_points_ms[-1] = t*1000
    # lasers and shutters in milliseconds:
    for tp in range(len(time_points)):
        if ABC_array[tp,0]==1: # laser A
            if tp < len(time_points_ms) - 1:
                laserA.append((time_points_ms[tp], time_points_ms[tp+1]))
                # if green laser on open Shutter orange:
                shutterOrange.append((time_points_ms[tp], time_points_ms[tp+1]))
            else:
                laserA.append((time_points_ms[tp], int(t*1000))) # final value
                shutterOrange.append((time_points_ms[tp], int(t*1000)))
        if ABC_array[tp,1]==1:  # laser B
            if tp < len(time_points_ms) - 1:
                laserB.append((time_points_ms[tp], time_points_ms[tp+1]))
                # if orange laser on open Shutter orange:
                shutterOrange.append((time_points_ms[tp], time_points_ms[tp+1]))
            else:
                laserB.append((time_points_ms[tp], int(t*1000))) # final value
                shutterOrange.append((time_points_ms[tp], int(t*1000)))
        if ABC_array[tp,2]==1:  # laser C
            if tp < len(time_points_ms) - 1:
                laserC.append((time_points_ms[tp], time_points_ms[tp+1]))
            else:
                laserC.append((time_points_ms[tp], int(t*1000))) # final value
                shutterOrange.append((time_points_ms[tp], int(t*1000)))                      
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
    
# %% spilt config file

    def split_and_write_configs(config_save_path, trigger_points, total_time_s, initials, n_chunks):
        chunk_duration_s = total_time_s / n_chunks
        chunk_duration_ms = int(chunk_duration_s * 1000)
    
        for i in range(n_chunks):
            start_ms = int(i * chunk_duration_s * 1000)
            end_ms = int((i + 1) * chunk_duration_s * 1000)
    
            # Prepare chunk-specific trigger intervals
            chunk_trigger_points = {}
            for key, intervals in trigger_points.items():
                chunk_intervals = []
                for s, e in intervals:
                    # Only include intervals that intersect with this chunk
                    if e > start_ms and s < end_ms:
                        # Clip and shift to start from 0 in this chunk
                        new_s = max(s, start_ms) - start_ms
                        new_e = min(e, end_ms) - start_ms
                        chunk_intervals.append((int(new_s), int(new_e)))
                        
                    if e==block_time_ms and s==block_time_ms: # for the always off 
                        new_s = chunk_duration_ms
                        new_e = chunk_duration_ms
                        chunk_intervals.append((int(new_s), int(new_e)))
                        
                chunk_trigger_points[key] = chunk_intervals
    
            # File name: add _part1, _part2, etc.
            base, ext = os.path.splitext(config_save_path)
            chunk_filename = f"{base}_part{i+1}{ext}"
    
            # Write the config file
            write_config_file(chunk_filename, chunk_trigger_points, chunk_duration_ms, initials)    
            


# %% plot trigger file like Anushka
    # Setting the figure size and resolution
    fig7 = plt.figure(figsize=(9, 4), dpi=300)
    frame_time_ms = int(frame_time*1000)
    framenumber=int(t/frame_time) # both in s
    ax = fig7.add_subplot(111)

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
# 0. Which figures to export
figures_to_save = [fig1, fig2, fig3, fig4, fig6, fig7]  # fig5,
# 1. Create dated directory
date_str = datetime.now().strftime("%Y%m%d")
dir_name = f"{date_str}_{trigger_name}"
full_save_path = os.path.join(save_path, dir_name)
os.makedirs(full_save_path, exist_ok=True)
    
print(f"Saving files to: {full_save_path}")

# 2. Save all figures as individual PNGs and combined PDF
pdf_path = os.path.join(full_save_path, f"{trigger_name}_figures.pdf")
with PdfPages(pdf_path) as pdf:
    for i, fig in enumerate(figures_to_save):
        if fig.get_suptitle():
            figtitle = fig.get_suptitle()
        else:
            figtitle = fig.axes[0].get_title()
        png_path = os.path.join(full_save_path, f"{trigger_name}_fig{i+1}_{figtitle}.png")
        fig.savefig(png_path)
        pdf.savefig(fig)
print("Figures saved.")

# 3. Save config file
# the whole laser config file
config_save_path = os.path.join(full_save_path, f"{trigger_name}_Laserconfig.txt")
write_config_file(config_save_path, trigger_pointsNEW, block_time_ms, initials)
# the laser config file in chunks:
if cut_it:
    split_and_write_configs(config_save_path, trigger_pointsNEW, t, initials, n_chunks)

# 4. Save extra input parameters to another txt
params_path = os.path.join(full_save_path, f"{trigger_name}_params.txt")
with open(params_path, "w") as f:
    f.write("Gillespie input Parameters:\n")
    f.write(f"population: \t{N}\n")
    f.write(f"Initial populations: \t{initials}\n")
    f.write("rates:\n")
    f.write(f"""alpha_forward =\t{alpha_for} s^-1\t rate for A->B
beta_forward =\t{beta_for} s^-1\t, rate for B->C
gamma_forward =\t{gamma_for} s^-1\t rate for C->A
alpha_backward =\t{alpha_back} s^-1\t rate for A<-B
beta_backward =\t{beta_back} s^-1\t rate for B<-C
gamma_backward =\t{gamma_back} s^-1\t rate for C<-A\n""")
    f.write("#\n")
    f.write("Calculated Hidden Markov Model:\n")
    f.write(f"Transition Matrix:\n{Transitionmatrix}\n")
    f.write(f"Steady State Population [probabilities]:\n{SteadyStatePi}\n")
    f.write(f"DeltaG: \t{DeltaG:.4f} kBT\n")
    f.write(f"DeltaS: \t{DeltaS:.4e} kBT\n")
    f.write("#\n")
    f.write("Trigger Settings:\n")
    f.write(f"Exposure Time: \t{exposure_time} s\n")
    f.write(f"Readout Time: \t{readout_time} s\n")
    f.write(f"Duration:\t{t} s")
    
print("Parameters file saved.")

# 5. Save state sequences
# in frames
sequence_path = os.path.join(full_save_path, f"{trigger_name}_StateSequences_frames.txt")
time_points_string = ', '.join(['{:.3f}'.format(i) if type(i) == float else str(i) for i in time_points])
header=f"""true time_points by Gillespie [s]
{time_points_string}
time[s]  \tStateSequence \tStateA \tStateB \tStateC"""
np.savetxt(sequence_path, np.c_[time_vector, state_array, A_array, B_array, C_array],
               header=header, fmt='%.2f %d %d %d %d',
               delimiter='\t')

print("State sequence [frames] file saved.")
# State sequence in ms for continous lasers
sequence_path = os.path.join(full_save_path, f"{trigger_name}_StateSequences_ms.txt")
time_points_string = ', '.join(['{:.3f}'.format(i) if type(i) == float else str(i) for i in time_points])
header=f"""true time_points by Gillespie [s]
{time_points_string}
time[s]  \tStateSequence \tStateA \tStateB \tStateC"""
np.savetxt(sequence_path, np.c_[time_vector_mslike, state_array_ms, A_array_ms, B_array_ms, C_array_ms],
               header=header, fmt='%.3f %d %d %d %d',
               delimiter='\t')

print("State sequence [ms] file saved.")