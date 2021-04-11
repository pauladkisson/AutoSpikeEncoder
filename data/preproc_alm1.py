#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:07:41 2021

@author: pauladkisson

Purpose: Pre-process anterior lateral motor cortex 1 (alm1) dataset from 
    Collaborative Research in Computational Neuroscience - Data Sharing
    (CRCNS; https://crcns.org/data-sets/motor-cortex/alm-1).
"""
import os
import shutil
from scipy.io import loadmat
import numpy as np
import tarfile
import pathlib
from crcnsget import download
from bs4 import BeautifulSoup
import requests
from SpikePreProcessor import SpikePreProcessor
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) #suppress max figure warning

def pull_data(url):
    '''
    Pulls data from url and returns the raw voltage recording.

    Parameters
    ----------
    url : str
        url of data.
    fsample : float
        sampling rate.

    Returns
    -------
    raw_data : list
        raw voltage recording. length of list is num_trials.  Each element is
        a ndarray (num_samples, num_channels).
    '''
    print("...Downloading...")
    username = "pauladkisson"
    password = "abcdefg"
    script_dir = os.path.dirname(__file__)
    download_path = os.path.join(script_dir, "temp.tar")
    download(url, username, password, "temp.tar")
    
    print("...Extracting...")
    tar = tarfile.open(download_path)
    tar.extractall()
    tar.close()
    foldername = url.split('/')[-1][:-4]
    extract_path = os.path.join(script_dir, foldername)
    new_extract_path = os.path.join(script_dir, "temp")
    shutil.rmtree(new_extract_path)
    os.rename(extract_path, new_extract_path)
    
    print("...Loading .mat files...")
    raw_data = []
    for file in os.listdir(new_extract_path):
        if file.startswith("."):
            continue
        else:
            file_path = os.path.join(new_extract_path, file)
            matdict = loadmat(file_path)
            ch_mua = matdict['ch_MUA']         
            raw_data.append(ch_mua)
    
    return raw_data

def get_urls():
    URL = 'https://portal.nersc.gov/project/crcns/download/index.php'
    login_data = dict(
            username="pauladkisson",
            password="abcdefg",
            fn="alm-1/datafiles/voltage_traces",
            submit='Login' 
            )
    with requests.Session() as s:
        r = s.post(URL,data=login_data,stream=True)
        with open("alm1_download_info.html", "wb") as f:
            f.write(r.content)
    with open("alm1_download_info.html", "r") as f:
        download_info = f.read()
    soup = BeautifulSoup(download_info, 'html.parser')
    urls = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href[-4:] == ".tar":
            alm1_idx = href.find("alm-1/")
            urls.append(href[alm1_idx:])
    return urls

fsample = 19531.25
num_channels = 32
preproc_alm1 = SpikePreProcessor(num_channels, fsample, vis=False)

urls = get_urls()

for session_num, url in enumerate(urls):
    print("Session %s / %s" %(session_num+1, len(urls)))
    raw_data = pull_data(url)
    print("...Processing...")
    for trial_num, trial_data in enumerate(raw_data):
        print("Trial %s / %s" % (trial_num+1, len(raw_data)))
        data = preproc_alm1(trial_data)
        normed_spikes, spike_times, normed_lfp, max_spike_voltages, max_lfp_voltages = data
        trial_pathname = "alm1/session_"+str(session_num)+"/trial_"+str(trial_num)
        trial_path = pathlib.Path(trial_pathname)
        trial_path.mkdir(parents=True, exist_ok=True)
        np.save(trial_pathname+"/lfp.npy", normed_lfp)
        np.save(trial_pathname+"/max_spike_voltages.npy", max_spike_voltages)
        np.save(trial_pathname+"/max_lfp_voltages.npy", max_lfp_voltages)
        for channel in range(num_channels):
            channel_pathname = trial_pathname+"/channel_"+str(channel)
            channel_path = pathlib.Path(channel_pathname)
            channel_path.mkdir(parents=True, exist_ok=True)
            np.save(channel_pathname+"/spikes.npy", normed_spikes[channel])
            np.save(channel_pathname+"/spike_times.npy", spike_times[channel])
            
#delete temps
script_dir = os.path.dirname(__file__)
temp_path = os.path.join(script_dir, "temp")
shutil.rmtree(temp_path)
temp_path = os.path.join(script_dir, "temp.tar")
os.remove(temp_path)
