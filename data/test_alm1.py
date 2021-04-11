#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 23:21:44 2021

@author: pauladkisson

Purpose: test saving and loading of alm1 .npy files
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import random

script_dir = os.path.dirname(__file__)
alm1_path = os.path.join(script_dir, "alm1")
session_folder = random.choice(os.listdir(alm1_path))
while session_folder.startswith("."):
    session_folder = random.choice(os.listdir(alm1_path))
session_path = os.path.join(alm1_path, session_folder)
print(session_folder)

trial_folder = random.choice(os.listdir(session_path))
while trial_folder.startswith("."):
    trial_folder = random.choice(os.listdir(session_path))
trial_path = os.path.join(session_path, trial_folder)
print(trial_folder)

lfp = np.load(os.path.join(trial_path, "lfp.npy"))
max_lfp_voltages = np.load(os.path.join(trial_path, "max_lfp_voltages.npy"))
max_spike_voltages = np.load(os.path.join(trial_path, "max_spike_voltages.npy"))

plt.figure()
plt.title("max_lfp_voltages")
plt.xlabel("Channel")
plt.plot(max_lfp_voltages)

plt.figure()
plt.title("max_spike_voltages")
plt.xlabel("Channel")
plt.plot(max_spike_voltages)

for channel_folder in os.listdir(trial_path):
    if channel_folder.endswith(".npy") or channel_folder.startswith("."):
        continue
    channel = int(channel_folder.split("_")[-1])
    channel_path = os.path.join(trial_path, channel_folder)
    spike_times = np.load(os.path.join(channel_path, "spike_times.npy"))
    spikes = np.load(os.path.join(channel_path, "spikes.npy"))
    
    plt.figure()
    plt.title("Spikes Channel %s" %channel)
    plt.plot(np.transpose(spikes))
    
    plt.figure()
    plt.title("Local Field Potential Channel %s" %channel)
    plt.plot(lfp[:, channel])
    
            