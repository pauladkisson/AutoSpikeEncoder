#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  18 16:07:41 2021

@author: pauladkisson

Purpose: Pre-process simulated dataset from Pedreira et al. (2012)
    (https://www135.lamp.le.ac.uk/hgr3/).
"""
import os
from scipy.io import loadmat
import numpy as np
import pathlib
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from SpikePreProcessor import SpikePreProcessor
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.max_open_warning": 0})  # suppress max figure warning


def download(file_url, filename):
    base_url = "https://www135.lamp.le.ac.uk/hgr3/"
    url = base_url + file_url
    r = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        chunk_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            if chunk:
                f.write(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()


def pull_data(url):
    """
    Pulls data from url and returns the raw voltage recording.

    Parameters
    ----------
    url : str
        url of data.

    Returns
    -------
    raw_data : ndarray (num_samples, num_channels)
        raw voltage recording
    """
    print("...Downloading...")
    script_dir = os.path.dirname(__file__)
    download_path = os.path.join(script_dir, "temp.mat")
    download(url, download_path)

    print("...Loading .mat file...")
    try:
        matdict = loadmat(download_path)
        raw_data = np.transpose(matdict["data"])
    except (ValueError, TypeError):  # .mat file is corrupted
        print("File is corrupted, so it will be skipped")

    return raw_data


def pull_ground_truth(url):
    """
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
    """
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
    soup = BeautifulSoup(download_info, "html.parser")
    urls = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href[-4:] == ".mat" and href != "ground_truth.mat":
            urls.append(href)
    return urls


# Constants
fsample = 24000
num_pts = 79
total_pts = fsample * 10 * 60
extract_lfp = False

urls = get_urls()
urls.sort(key=lambda url: int(url.split("_")[-1].split(".")[0]))
start_session = 0  # starting on a specific session number (index)
urls = urls[start_session:]

# Ground Truth
ground_truth = pull_ground_truth("ground_truth.mat")
# omitting spikes within num_pts of end
num_skip = []
for gt_session in ground_truth["spike_first_sample"][0]:
    num_skip_session = []
    for channel_spike_indices in gt_session:
        num_skip_channel = 0
        i = -1
        while (
            channel_spike_indices[i] + num_pts >= total_pts
        ):  # spike occurs within num_pts of end
            num_skip_channel += 1
            i -= 1
        if num_skip_channel != 0:
            num_skip_session.append(num_skip_channel)
    num_skip.append(num_skip_session)

for session, num_skip_session in enumerate(num_skip):
    for channel, num_skip_channel in enumerate(num_skip_session):
        ground_truth["spike_first_sample"][0, session] = ground_truth[
            "spike_first_sample"
        ][0, session][channel : channel + 1, :-num_skip_channel]
        ground_truth["spike_classes"][0, session] = ground_truth["spike_classes"][
            0, session
        ][channel : channel + 1, :-num_skip_channel]
# save results
gt_pathname = "pedreira/ground_truth"
gt_path = pathlib.Path(gt_pathname)
gt_path.mkdir(parents=True, exist_ok=True)
for field, val in ground_truth.items():
    if field.startswith("__"):
        continue
    field_pathname = gt_pathname + "/" + field
    np.save(field_pathname, val)
spike_first_sample = ground_truth["spike_first_sample"]
spike_classes  = ground_truth["spike_classes"]

# Simulated Data
for session_num, url in enumerate(urls):
    print(
        "Session %s / %s" % (session_num + start_session + 1, len(urls) + start_session)
    )
    raw_data = pull_data(url)
    print("...Processing...")
    num_channels = raw_data.shape[1]
    gt = spike_first_sample[0, session_num + start_session]
    gt_classes = spike_classes[0, session_num + start_session]
    preproc_pedreira = SpikePreProcessor(
        num_channels, fsample, vis=False, gt=gt, gt_classes=gt_classes, num_pts=num_pts, extract_lfp=extract_lfp
    )
    data = preproc_pedreira(raw_data)
    normed_spikes, spike_times, normed_lfp, max_spike_voltages, max_lfp_voltages, snrs = data
    session_pathname = "pedreira/session_" + str(session_num + start_session)
    session_path = pathlib.Path(session_pathname)
    session_path.mkdir(parents=True, exist_ok=True)
    if extract_lfp:
        np.save(session_pathname + "/lfp.npy", normed_lfp)
        np.save(session_pathname + "/max_lfp_voltages.npy", max_lfp_voltages)
    np.save(session_pathname + "/max_spike_voltages.npy", max_spike_voltages)
    np.save(session_pathname + "/snrs.npy", snrs)
    for channel in range(num_channels):
        channel_pathname = session_pathname + "/channel_" + str(channel)
        channel_path = pathlib.Path(channel_pathname)
        channel_path.mkdir(parents=True, exist_ok=True)
        np.save(channel_pathname + "/spikes.npy", normed_spikes[channel])
        np.save(channel_pathname + "/spike_times.npy", spike_times[channel])
    if session_num==1:
        break
        
# delete temps
script_dir = os.path.dirname(__file__)
download_path = os.path.join(script_dir, "temp.mat")
try:
    os.remove(download_path)
except FileNotFoundError:
    pass
