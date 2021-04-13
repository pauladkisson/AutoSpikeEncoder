#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:09:32 2021

@author: pauladkisson    
"""

from scipy.signal import iirfilter, lfilter, decimate, find_peaks
import matplotlib.pyplot as plt
import numpy as np

class SpikePreProcessor:
    '''
    Class for pre-processing raw voltage recordings
    '''
    
    def __init__(self, num_channels, fsample, thresh_factor=100, vis=False):
        self.num_channels = num_channels
        self.fs = fsample
        self.thresh_factor = thresh_factor
        self.lfp_fs = 200 #Nyquist rate for 100Hz
        self.downsample_factor = int(fsample / self.lfp_fs)
        self.spike_width = 1
        self.Ts = 1 / self.fs
        self.align_radius = 1*10**(-3)
        self.vis = vis
    
    def __call__(self, raw_data, vis=None):
        '''
        lowpass filter raw voltage recordings and downsample to obtain LFPs,
        bandpass filter raw voltage recordings,
        threshold to detect spikes,
        extract 2ms spike windows,
        align spikes,
        normalize to (-1, 1)
        if vis, visualize results

        Parameters
        ----------
        raw_data : ndarray (num_samples, num_channels)
            raw extracellular electrode recording

        Returns
        -------
        normed_spikes : list (num_channels)
            list of ndarrays of aligned 2ms spikes (num_spikes, num_timesteps)
            normalized into range (-1, 1)
        spike_times : list (num_channels)
            list of ndarrays of spike times (num_spikes,)
        normed_lfp : ndarray (reduced_num_samples, num_channels)
            local field potential downsampled to lfp_fs Hz, normalized to (-1, 1)
        max_spike_voltages : ndarray (num_channels,)
            maximum absolute voltage of each channel used for spike normalization
        max_lfp_voltages : ndarray (num_channels,)
            maximum absolute voltage of each channel used for spike normalization

        '''
        if vis is None:
            vis = self.vis
        bp_data, lfp = self.filter_data(raw_data)
        spike_indices = self.threshold(bp_data)
        aligned_spikes, spike_times = self.align(bp_data, spike_indices)
        normed_spikes, normed_lfp, max_spike_voltages, max_lfp_voltages = self.normalize(aligned_spikes, lfp)
        if vis:
            self.visualize(raw_data, normed_spikes, spike_times, normed_lfp, max_spike_voltages, max_lfp_voltages)
        return normed_spikes, spike_times, normed_lfp, max_spike_voltages, max_lfp_voltages
    
    def filter_data(self, raw_data):
        '''
        bandpass filter to raw voltage recordings, and lowpass filter + 
        decimate to obtain local field potentials.

        Parameters
        ----------
        raw_data : ndarray (num_samples, num_channels)
            raw extracellular electrode recording

        Returns
        -------
        bp_data : ndarray (num_samples, num_channels)
            bandpass filtered voltage recordings
        lfp_data : ndarray (reduced_num_samples, num_channels)
            local field potential downsampled to lfp_fs Hz

        '''
        
        bp_cutoffs = (300, 3000)
        lfp_cutoff = 100
        b, a = iirfilter(1, bp_cutoffs, fs=self.fs)
        bp_data = lfilter(b, a, raw_data, axis=0)
        b, a = iirfilter(1, lfp_cutoff, fs=self.fs, btype="lowpass")
        lfp_data = decimate(raw_data, self.downsample_factor, axis=0)
        lfp_data = lfilter(b, a, lfp_data, axis=0)
        
        return bp_data, lfp_data
    
    def neo(self, x):
        '''
        Non-linear Energy Operator (NEO)
        psi(n) = x(n)^2 - x(n-1)x(n+1)
        Boundary conditions:
            psi(0) = x(0)^2 - x(0)x(1),
            psi(N) = x(N)^2 - x(N)x(N-1)

        Parameters
        ----------
        x : ndarray
            Input (vector or matrix).

        Returns
        -------
        psi : ndarray
            Output (same shape as x).

        '''
        psi = np.zeros(x.shape)
        psi[0] = x[0]**2 - x[0]*x[1]
        psi[-1] = x[-1]**2 - x[-1]*x[-2]
        psi[1:-1] = x[1:-1]**2 - x[2:]*x[:-2]
        return psi
    
    def threshold(self, data):
        '''
        Threshold data.

        Parameters
        ----------
        data : ndarray (num_samples, num_channels)
            bandpass filtered voltage recordings

        Returns
        -------
        spike_indices : list (num_channels)
            list of ndarrays of indices (num_spikes,) denoting time of each spike

        '''
        data = self.neo(data)
        sigma = np.median(np.abs(data), axis=0) / 0.67
        thresh = self.thresh_factor * sigma
        
        spike_indices = []
        for channel in range(self.num_channels):
            peak_indices, _ = find_peaks((data[:, channel]), thresh[channel], width=self.spike_width)
            spike_indices.append(peak_indices)
        return spike_indices
    
    def align(self, data, spike_indices):
        '''
        align spikes to the absolute maxima within the 2ms window, extract a
        new 2ms window around the aligned spike times.

        Parameters
        ----------
        data : ndarray (num_samples, num_channels)
            bandpass filtered voltage recordings
        spike_indices : list (num_channels)
            list of ndarrays of indices (num_spikes,) denoting time of each spike

        Returns
        -------
        aligned_spikes : list (num_channels)
            list of ndarrays (num_spikes, num_samples) containing 2ms aligned
            spikes.
        spike_times : list (num_channels)
            list of ndarrays (num_spikes,) containing spike times

        '''
        tmax = self.Ts*(data.shape[0]-1)
        t = np.linspace(0, tmax, data.shape[0])
        aligned_spikes = []
        spike_times = []
        for channel in range(self.num_channels):
            try:
                old_channel_spike_times = t[spike_indices[channel]]
            except IndexError: #No spikes detected
                assert len(spike_indices[channel]) == 0
                old_channel_spike_times = []
            
            channel_spikes = []
            channel_spike_times = []
            for i, spike_time in enumerate(old_channel_spike_times):
                #obtain raw spike
                spike_window = (spike_time - self.align_radius, spike_time + self.align_radius)
                if spike_window[0] < 0 or spike_window[1] > tmax: #spike occurs within 1ms of start or end
                    continue
                window_mask = np.logical_and(t >= spike_window[0], t <= spike_window[1])
                t_spike = t[window_mask]
                spike = data[window_mask, channel]
                
                #align spike to argmax(diff(spike))
                new_spike_idx = np.argmax(np.diff(spike))
                new_spike_time = t_spike[new_spike_idx]
                new_spike_window = (new_spike_time - self.align_radius, new_spike_time + self.align_radius)
                if new_spike_window[0] < 0 or new_spike_window[1] > tmax: #aligned spike occurs within 1ms of start or end
                    continue
                new_window_mask = np.logical_and(t >= new_spike_window[0], t <= new_spike_window[1])
                aligned_spike = data[new_window_mask, channel]
                
                #record
                if np.isin(new_spike_time, np.array(channel_spike_times)): #spike has already been detected
                    continue
                else:
                    channel_spikes.append(aligned_spike)
                    channel_spike_times.append(new_spike_time)
            
            aligned_spikes.append(np.array(channel_spikes))
            spike_times.append(np.array(channel_spike_times))
        
        return aligned_spikes, spike_times
        
    def normalize(self, aligned_spikes, lfp):
        '''
        min-max normalize into range (-1, 1) for easier encoding

        Parameters
        ----------
        aligned_spikes : list (num_channels)
            list of ndarrays (num_spikes, num_samples) containing 2ms aligned
            spikes.
        lfp : ndarray (reduced_num_samples, num_channels)
            local field potential downsampled to lfp_fs Hz

        Returns
        -------
        normed_spikes : list (num_channels)
            aligned_spikes normalized into (-1, 1)
        normed_lfp : ndarray (reduced_num_samples, num_channels)
            lfp normalized into (-1, 1)
        max_spike_voltages : ndarray (num_channels,)
            maximum absolute voltage of each channel used for spike normalization
        max_lfp_voltages : ndarray (num_channels,)
            maximum absolute voltage of each channel used for spike normalization

        '''
        normed_spikes = []
        max_spike_voltages = []
        for channel in range(self.num_channels):
            channel_spikes = aligned_spikes[channel]
            try:
                max_voltage = np.max(np.abs(channel_spikes))
                normed_spikes.append(channel_spikes / max_voltage)
            except ValueError: #No spikes detected
                assert len(channel_spikes) == 0
                max_voltage = np.nan
                normed_spikes.append(np.array([]))
            max_spike_voltages.append(max_voltage)
            
        max_lfp_voltages = np.max(np.abs(lfp), axis=0)
        dead_channel_mask = max_lfp_voltages==0
        max_lfp_voltages[dead_channel_mask] = 1 #prevent division by 0 for dead channels
        normed_lfp = lfp / max_lfp_voltages
        max_lfp_voltages[dead_channel_mask] = 0 #recover dead channels
        
        return normed_spikes, normed_lfp, np.array(max_spike_voltages), max_lfp_voltages
    
    def visualize(self, raw_data, normed_spikes, spike_times, normed_lfp, max_spike_voltages, max_lfp_voltages):
        '''
        visualize results of preprocessing:
            spike trains, normalized voltage recordings,
            normalized aligned spikes, and normalized LFPs

        Parameters
        ----------
        raw_data : ndarray (num_samples, num_channels)
            raw extracellular electrode recording
        normed_spikes : list (num_channels)
            aligned_spikes normalized into (-1, 1)
        spike_times : list (num_channels)
            list of ndarrays (num_spikes,) containing spike times
        normed_lfp : ndarray (reduced_num_samples, num_channels)
            lfp normalized into (-1, 1)
        max_spike_voltages : ndarray (num_channels,)
            maximum absolute voltage of each channel used for spike normalization
        max_lfp_voltages : ndarray (num_channels,)
            maximum absolute voltage of each channel used for spike normalization

        Returns
        -------
        None.

        '''
        num_samples = raw_data.shape[0]
        tmax = self.Ts * (num_samples-1)
        t = np.linspace(0, tmax, num_samples)
        align_t = np.arange(-1*self.align_radius, self.align_radius - self.Ts, self.Ts)*1000
        lfp_t = np.linspace(0, tmax, normed_lfp.shape[0])
        
        plt.figure()
        plt.plot(max_spike_voltages)
        plt.title("Max spike voltages used for normalization")
        plt.xlabel("Channel")
        plt.ylabel("Voltage")
        
        plt.figure()
        plt.plot(max_lfp_voltages)
        plt.title("Max local field potentials used for normalization")
        plt.xlabel("Channel")
        plt.ylabel("Voltage")
        
        for channel in range(self.num_channels):
            if np.all(raw_data[:, channel] == 0): #dead channel
                print("Channel %s is dead." %channel)
                continue
            normed_raw_data = raw_data[:, channel] / np.max(np.abs(raw_data[:, channel]))
            spike_train = np.zeros((num_samples,))
            spike_train[np.isin(t, spike_times[channel])] = 1
            assert spike_train.sum() == spike_times[channel].size, \
"Incorrect spike train construction: spike train should have %s spikes but has %s spikes instead" % (spike_times[channel].size, spike_train.sum())

            plt.figure()
            plt.hold = True
            plt.plot(t, spike_train+2, label="Multi-Unit Spike Train")
            plt.plot(t, normed_raw_data, label="Normalized Raw Electrode Recording")
            plt.hold = False
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized Voltage")
            plt.title("Detected Spikes : Channel %s" %channel)
            plt.legend(loc="center")
        
            plt.figure()
            plt.plot(lfp_t, normed_lfp[:, channel])
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.title("Local Field Potential : Channel %s" %channel)
            
            if normed_spikes[channel].size==0:
                print("Channel %s did not detect any spikes." %channel)
                continue
            plt.figure()
            plt.plot(align_t, np.transpose(normed_spikes[channel]))
            plt.xlabel("Time (ms)")
            plt.ylabel("Normalized Voltage")
            plt.title("Aligned Spikes : Channel %s" %channel)
                
                       