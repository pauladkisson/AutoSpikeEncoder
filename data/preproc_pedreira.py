#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  18 16:07:41 2021

@author: pauladkisson

Purpose: Pre-process simulated dataset from Pedreira et al. (2012)
    (https://www135.lamp.le.ac.uk/hgr3/).
"""
import os
import shutil
from scipy.io import loadmat
import numpy as np
import tarfile
import pathlib
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from SpikePreProcessor import SpikePreProcessor
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) #suppress max figure warning

def download(file_url, filename):
    base_url = "https://www135.lamp.le.ac.uk/hgr3/"
    url = base_url+file_url
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        total_size_in_bytes= int(r.headers.get('content-length', 0))
        chunk_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            if chunk:
                f.write(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()
    

def pull_data(url):
    '''
    Pulls data from url and returns the raw voltage recording.

    Parameters
    ----------
    url : str
        url of data.

    Returns
    -------
    raw_data : ndarray (num_samples, num_channels)
        raw voltage recording
    '''
    print("...Downloading...")
    script_dir = os.path.dirname(__file__) 
    download_path = os.path.join(script_dir, "temp.mat")
    download(url, download_path)
    
    print("...Loading .mat file...")
    try:
        matdict = loadmat(download_path)
        raw_data = np.transpose(matdict['data'])
    except (ValueError, TypeError): #.mat file is corrupted
        print("File is corrupted, so it will be skipped")
    
    return raw_data

def pull_ground_truth(url):
    '''
    Pulls ground truth data from url and returns ground truth fields:
        su_waveforms, spike_classes, and spike_first_sample (see Readme.doc) 

    Parameters
    ----------
    url : str
        url of data.

    Returns
    -------
    ground_truth : dictionary
        ground truth of simulation.
    '''
    print("...Downloading...")
    script_dir = os.path.dirname(__file__) 
    download_path = os.path.join(script_dir, "temp.mat")
    download(url, download_path)
    
    print("...Loading .mat file...")
    ground_truth = loadmat(download_path)
    
    return ground_truth

def get_urls():
    URL = "https://www135.lamp.le.ac.uk/hgr3/"
    r = requests.get(URL)
    with open("pedreira_download_info.html", "wb") as f:
        f.write(r.content)
    with open("pedreira_download_info.html", "r") as f:
        download_info = f.read()
    soup = BeautifulSoup(download_info, 'html.parser')
    urls = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href[-4:] == ".mat" and href != "ground_truth.mat":
            urls.append(href)
    return urls

#Constants
fsample = 24000
num_pts = 79

urls = get_urls()
start_session = 0 #starting on a specific session number (index)
urls = urls[start_session:]

#Ground Truth
ground_truth = pull_ground_truth("ground_truth.mat")
gt_pathname = "pedreira/ground_truth"
gt_path = pathlib.Path(gt_pathname)
gt_path.mkdir(parents=True, exist_ok=True)
for field, val in ground_truth.items():
    if field.startswith("__"):
        continue
    field_pathname = gt_pathname+"/"+field
    np.save(field_pathname, val)
spike_first_sample = ground_truth["spike_first_sample"]

#Simulated Data
for session_num, url in enumerate(urls):
    print("Session %s / %s" %(session_num+start_session+1, len(urls)+start_session))
    raw_data = pull_data(url)
    print("...Processing...")
    num_channels = raw_data.shape[1]
    gt = spike_first_sample[0, session_num]
    preproc_pedreira = SpikePreProcessor(num_channels, fsample, vis=False, gt=gt, num_pts=num_pts)
    data = preproc_pedreira(raw_data)
    normed_spikes, spike_times, normed_lfp, max_spike_voltages, max_lfp_voltages = data
    session_pathname = "pedreira/session_"+str(session_num+start_session)
    session_path = pathlib.Path(session_pathname)
    session_path.mkdir(parents=True, exist_ok=True)
    np.save(session_pathname+"/lfp.npy", normed_lfp)
    np.save(session_pathname+"/max_spike_voltages.npy", max_spike_voltages)
    np.save(session_pathname+"/max_lfp_voltages.npy", max_lfp_voltages)
    for channel in range(num_channels):
        channel_pathname = session_pathname+"/channel_"+str(channel)
        channel_path = pathlib.Path(channel_pathname)
        channel_path.mkdir(parents=True, exist_ok=True)
        np.save(channel_pathname+"/spikes.npy", normed_spikes[channel])
        np.save(channel_pathname+"/spike_times.npy", spike_times[channel])
            
#delete temps
script_dir = os.path.dirname(__file__)
download_path = os.path.join(script_dir, "temp.mat")
try:
    os.remove(download_path)
except FileNotFoundError:
    pass