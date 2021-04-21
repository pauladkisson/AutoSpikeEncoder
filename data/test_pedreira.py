#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:21:44 2021

@author: pauladkisson

Purpose: test saving and loading of pedreira .npy files
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import random

script_dir = os.path.dirname(__file__)
pedreira_path = os.path.join(script_dir, "pedreira")
session_folder = random.choice(os.listdir(pedreira_path))
while session_folder.startswith(".") or session_folder=="ground_truth" or session_folder.endswith(".doc"):
    session_folder = random.choice(os.listdir(pedreira_path))
session_path = os.path.join(pedreira_path, session_folder)
print(session_folder)

lfp = np.load(os.path.join(session_path, "lfp.npy"))
max_lfp_voltages = np.load(os.path.join(session_path, "max_lfp_voltages.npy"))
max_spike_voltages = np.load(os.path.join(session_path, "max_spike_voltages.npy"))

plt.figure()
plt.title("max_lfp_voltages")
plt.xlabel("Channel")
plt.plot(max_lfp_voltages)

plt.figure()
plt.title("max_spike_voltages")
plt.xlabel("Channel")
plt.plot(max_spike_voltages)

for channel_folder in os.listdir(session_path):
    if channel_folder.endswith(".npy") or channel_folder.startswith("."):
        continue
    channel = int(channel_folder.split("_")[-1])
    channel_path = os.path.join(session_path, channel_folder)
    spike_times = np.load(os.path.join(channel_path, "spike_times.npy"))
    spikes = np.load(os.path.join(channel_path, "spikes.npy"))
    
    plt.figure()
    plt.title("Spikes Channel %s" %channel)
    plt.plot(np.transpose(spikes))
    
    plt.figure()
    plt.title("Local Field Potential Channel %s" %channel)
    plt.plot(lfp[:, channel])
    
            