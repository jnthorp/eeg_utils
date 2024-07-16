# -*- coding: utf-8 -*-

from glob import glob
import os
from collections import OrderedDict

from mne import create_info, concatenate_raws
from mne.io import RawArray
from mne.io import Raw
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import mne #hxf 2/27/2021
if mne.__version__<='0.17.0':
    from mne.channels import read_montage #for mne version 0.17.0
else:
    from mne.channels import make_standard_montage  # hxf 2/27/2021, for mne verion 0.22.0
from scipy import signal
from scipy.integrate import simps
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


sns.set_context('talk')
sns.set_style('white')

def get_filePath(cleaned = False):
    '''If passed cleaned as true, get all file path with _cleaned in file name'''
    '''returns a list of directory of files'''
    '''default place to call this function is for py files at the EEG folder level'''
    file_dir = []
    subjects = [i for i in sorted(os.listdir('..' + os.sep)) if 'sub-' in i and 'instructor' not in i]
    for subj in subjects:
        sessions = [i for i in sorted(os.listdir('..' + os.sep + subj)) if 'ses-' in i]
        for ses in sessions:
            data_dir = '..' + os.sep + subj + os.sep + ses
            files = []
            if cleaned:
                files = [i for i in sorted(os.listdir(data_dir)) if '_cleaned.csv' in i]
            else:
                files = [i for i in sorted(os.listdir(data_dir)) if '_cleaned.csv' not in i]
            for file in files:
                file_dir.append('..' + os.sep + subj + os.sep + ses + os.sep + file)
    return file_dir

def subtract_time(reset_ind,data,reset_times):
    '''Subtract time from the reset times to the end of the sawtooth'''
    '''Only necessary for Summer 2023 data'''
    #row.name is the reset index ([0,1,2,...])
    #row.times is the index in data ([50087, 61382, ...])
    for row in reset_ind.iterrows():
        
        row_name = row[0]
        row_times = list(row[1])[0]
        
        here = row_times
        start = row_times
        end = reset_ind['times'][row_name - 1] if row_name > 0 else reset_ind.times[0]
        
        while here > end:
            data.at[here, 'converted_timestamp'] = reset_times['converted_timestamp'][start] - ((1/256) * (start - here))
            here -= 1
    return data
    
def sawtooth():
    '''Clean the sawtoothed EEG data'''
    '''Summer 2023 data only'''
    file_dir = get_filePath(cleaned = False)
    for file in file_dir:
        try:
            # read in file as pandas data frame
            data = pd.read_csv(file)
            
            # convert timestamps to check negativity
            data['converted_timestamp'] = data['timestamps'] - data['timestamps'][0]
            
            # get the diffs in timestamps 1, 2, and 3 rows away
            data['diff1'] = data['timestamps'].diff(1)
            data['diff2'] = data['timestamps'].diff(2)
            data['diff3'] = data['timestamps'].diff(3)
            
            # label rows as descending (wrong, should be thrown away) if all three rows are less than 0
            data['descending'] = (data['diff1'] < 0) & (data['diff2'] < 0) & (data['diff3'] < 0)
            
            if True in data['descending'].tolist():
                # filter out rows within descending periods
                data = data[~data['descending']]
                data.reset_index(drop=True, inplace=True)
                
                # a reset time is when a forward sawtooth is ascending (diff 2 to diff 3) and then jumps back down (diff 1 < 0)
                reset_times = pd.DataFrame({'converted_timestamp': data['converted_timestamp'][(data['diff1'] < 0) 
                                                                                               & data['diff3'] > data['diff2']]})
                
                # assign the reset time back one index
                reset_times.index = reset_times.index - 1
                reset_times['diff1'] = reset_times['converted_timestamp'].diff()
                
                # make the timestamp the next sawtooth started at the timestamp the last sawtooth ended at
                data.update(reset_times)
                reset_ind = pd.DataFrame({'times': reset_times.index})
                
                # Make the timestamps lead up to the end of each sawtooth at 256 Hz
                data = subtract_time(reset_ind,data,reset_times)

                # drop everything after the last reset
                last = reset_ind.times.iloc[-1]
                data.drop(data[last:].index, inplace=True)
                
                # drop duplicate datapoints
                data.drop_duplicates(subset=['AF7','AF8','TP9','TP10'], inplace=True)
                data.reset_index(drop=True, inplace=True)

                # write all this to timestamps
                data['timestamps'] = data['timestamps'][0] + data['converted_timestamp']
                
                #look for any skipsof greater than 10 seconds so we can throw out any chunks less than 10 seconds long (2560 samples)
                skips = pd.Series(data.loc[abs(data['timestamps'].diff(1)) > 10].index)
                skips = pd.concat([skips, pd.Series(data.shape[0])]).reset_index(drop=True)
                skips_diff = skips.diff()
                skips_diff.index = skips_diff.index - 1
                skip_idx = [i for i in skips_diff.index if skips_diff[i] < 2560]
                #print(skips, skips_diff, skip_idx)
                for i in skip_idx:
                    drop_idx = range(skips[i], skips[i+1])
                #    print(drop_idx)
                    data.drop(drop_idx, inplace=True)

            data = data[['timestamps','TP9','AF7','AF8','TP10','Right AUX','Marker0']]
                
            # write to a new file
            new_filename = file.split('.csv')[0] + '_cleaned.csv'
            data.to_csv(new_filename, index=False)
                
        except Exception as error:
            print(error)

def compileBehavior(questions, data_ses, short_ses, marker_lst, remove = True):
    '''Grab the markers from the instructor data and line them up with the questions'''
    
    # instructor_df is a df of what onsets for slide numbers
    instructor_dir = '../sub-instructor' + os.sep + data_ses
    instructor_files = [i for i in sorted(os.listdir(instructor_dir)) if '.csv' in i]
    instructor_df = pd.read_csv(instructor_dir + os.sep + instructor_files[0])
    
    # questions_ses is what slide numbers line up with what later questions
    questions_ses = questions.loc[questions['Week'] == int(short_ses),].reset_index(drop=True)
    
    behavior_df = pd.concat([instructor_df, questions_ses], axis=1)
    
    # Marker0 is the column for labels of what's going at that moment
    behavior_df['Marker0'] = behavior_df['Period']
    
    if remove:
        behavior_df = removeMarker_Behav(behavior_df, marker_lst)
        
    return behavior_df            

def removeMarker_Behav(behavior_df, marker_lst):
    ''' removes the marker that are not in the list marker_lst from the behavior_df 
    '''
    
    orig = list(behavior_df['Marker0'])
    
    bool_lst = []
    for marker in orig:
        if marker in marker_lst:
            bool_lst.append(False)
        else:
            bool_lst.append(True)
            
    behavior_df.Marker0.loc[bool_lst] = 0
    behavior_df = behavior_df[['timestamps','Marker0']]
    
    return behavior_df

def readRawAndProcess(data_fnames, behavior_df, filt = [0.1, 50, 60], sfreq = 256, plot = True):
    
    ''' read in a raw file based on sfreq
    '''
    
    l_freq = filt[0]
    h_freq = filt[1]
    freqs = filt[2]
    
    raw = load_muse_csv_as_raw(data_fnames, sfreq=sfreq, ch_ind=[0,1,2,3], fill=True, markers=behavior_df)
    regexp = r'(TP9|AF7|AF8|TP10| .)'
    artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
    filt_raw = raw.copy()
            
    filt_raw = filt_raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', 
                               skip_by_annotation='bad_acq_skip').notch_filter(freqs=freqs)
            
    ica = ICA(max_iter='auto', random_state=97)
    ica.fit(filt_raw, picks=filt_raw.ch_names[:-1])
            
    ica.exclude=[0] #ica component to exclude
    ica.apply(filt_raw)
    
    if plot:
        filt_raw.plot(scalings=100e-6)
    return filt_raw

def eventsPreprocess(events, durations, filt_raw, small_dur = 0, remove = True):
    events = np.column_stack((events, durations))
    events = events[events[:,3] != events[:,0]]
    events[:,2] = 999
    events_new = np.array([events[:,3],np.ones(events.shape[0])*999, np.zeros(events.shape[0])]).T
    events = events[:,:-1]
    events = np.row_stack((events, events_new))
    event_onsets = mne.find_stim_steps(filt_raw)
    events = np.row_stack((event_onsets, events))
    events = events[events[:, 0].argsort()]
    
    if remove:
        events = removeSmallDuration(events, small_dur)
    
    return events

def removeSmallDuration(events, val):
    small_diffs = np.diff(events[:,0],prepend=0) < 8

    small_diffs_loc = [i for i, x in enumerate(small_diffs) if x]
    for i in small_diffs_loc:
        events[i,1] = max((events[i,1], events[i-1,1]))
        events[i,2] = max((events[i,2], events[i-1,2]))

    small_diffs_drop = [i - 1 for i in small_diffs_loc]
    events = np.delete(events, small_diffs_drop, axis = 0)
    return events
    
def annotateRaw(events, filt_raw, sfreq = 256, plot = True):
    annotations = mne.annotations_from_events(events, sfreq)
    durations = (np.diff(events[:,0]) - 2) / sfreq
    durations = np.append(durations, 0)
    annotations.duration = durations  
            
    filt_raw = filt_raw.set_annotations(annotations)
    if plot:
        filt_raw.plot(scalings=100e-6)
    return filt_raw.crop_by_annotations()
    

def filtEpochDropBad(filt_raw, thresh, avg_dict = dict(AF=[1,2], TP=[0,3]), drop = True):
    question_number = filt_raw.annotations.description[0]

    # combine the channels into an AF and TP
    filt_raw = mne.channels.combine_channels(filt_raw, 
                                             groups=avg_dict, method='mean', keep_stim=True)

    # scale the threshold into microvolts
    thresh_scaled = (thresh*1e-6)

    # get the number of samples before this filtering step
    pre_filt_length = filt_raw.n_times

    # Separate the raw into 0.5 second long with a 250 ms overlap
    length = 0.5
    event_1s = [i for i in range(filt_raw.first_samp,filt_raw.first_samp + filt_raw.n_times) if 
                (i % int(round(filt_raw.info['sfreq'])*(length*3/4)) == 0)]
    events = np.array((event_1s, np.zeros((int(len(event_1s)))), np.ones(len(event_1s))), dtype='int')
    events = np.transpose(events)

    filt_raw = mne.Epochs(filt_raw, events, tmin=-1*length/2, tmax=length/2, 
                          reject=dict(eeg=thresh_scaled), flat=dict(eeg=1.0e-6), baseline=None)
    # make settable, default being drop
    if drop:
        filt_raw.drop_bad(verbose = False)
    
    return (filt_raw, pre_filt_length, question_number)    

def power(data, sfreq, col_list, params, range_val = 2):
    
    band_power, data_subject,data_ses,pre_filt_length,filtered_length, question_number = params
    
    band_power_lst = []
    for idx in range(range_val):
        arr = data[[col_list[idx]]].squeeze()
        
        win = 4 * sfreq
        freqs, psd = signal.welch(arr, sfreq, nperseg=win)
        
        bands = {"Delta":(0.5, 4,),"Theta":(4,8),"Alpha":(8,12),"Beta":(12,30),"Gamma":(30,50)}
        i = 1
        rows=2
        columns=3        
        
        for band, freq_range in bands.items():
            idx_band = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
            
            i = i + 1
            
            freq_res = freqs[1] - freqs[0]
            power = simps(psd[idx_band], dx=freq_res)
            print(col_list[idx] + ' - ' + band)
            print('Absolute ' + band + ' power: ' + "{:f}".format(power) + ' uV^2')
            
            total_power = simps(psd, dx=freq_res)
            
            rel_power = power / total_power
            print('Relative ' + band + ' power: '+ "{:f}".format(rel_power))
            
            welch = bandpower(arr, sfreq, freq_range, method='welch', window_sec=4, relative=False)
            
            band_power.loc[len(band_power.index)]=[col_list[idx],band,power,rel_power,welch,
                                                   data_subject,data_ses,pre_filt_length,filtered_length, question_number]
        print('-------------------------------------------------')
        print('\n')
    return pd.DataFrame(band_power)



def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp



def plot_spectrum_methods(data, sf, window_sec, band=None, dB=False):
    """Plot the periodogram, Welch's and multitaper PSD.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds for Welch's PSD
    dB : boolean
        If True, convert the power to dB.
    """
    from mne.time_frequency import psd_array_multitaper
    from scipy.signal import welch, periodogram
    sns.set(style="white", font_scale=1.2)
    # Compute the PSD
    freqs, psd = periodogram(data, sf)
    freqs_welch, psd_welch = welch(data, sf, nperseg=window_sec*sf)
    psd_mt, freqs_mt = psd_array_multitaper(data, sf, adaptive=True,
                                            normalization='full', verbose=0)
    sharey = False

    # Optional: convert power to decibels (dB = 10 * log10(power))
    if dB:
        psd = 10 * np.log10(psd)
        psd_welch = 10 * np.log10(psd_welch)
        psd_mt = 10 * np.log10(psd_mt)
        sharey = True

    # Start plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=sharey)
    # Stem
    sc = 'slategrey'
    ax1.stem(freqs, psd, linefmt=sc, basefmt=" ", markerfmt=" ")
    ax2.stem(freqs_welch, psd_welch, linefmt=sc, basefmt=" ", markerfmt=" ")
    ax3.stem(freqs_mt, psd_mt, linefmt=sc, basefmt=" ", markerfmt=" ")
    # Line
    lc, lw = 'k', 2
    ax1.plot(freqs, psd, lw=lw, color=lc)
    ax2.plot(freqs_welch, psd_welch, lw=lw, color=lc)
    ax3.plot(freqs_mt, psd_mt, lw=lw, color=lc)
    # Labels and axes
    ax1.set_xlabel('Frequency (Hz)')
    if not dB:
        ax1.set_ylabel('Power spectral density (V^2/Hz)')
    else:
        ax1.set_ylabel('Decibels (dB / Hz)')
    ax1.set_title('Periodogram')
    ax2.set_title('Welch')
    ax3.set_title('Multitaper')
    if band is not None:
        ax1.set_xlim(band)
    ax1.set_ylim(ymin=0)
    ax2.set_ylim(ymin=0)
    ax3.set_ylim(ymin=0)
    sns.despine()
    
    
    
## concatenate_and_fill_raws taken from the internet by John - concatenates raws and fills in dropped spots with 0's and "BAD_ACQ_SKIP" annotation

def concatenate_and_fill_raws(mne_raws: list, starts, ends) -> mne.io.RawArray:
    if type(mne_raws) is list:

        info = mne_raws[0].info

        # Check that filling gaps is possible
        for iRec, raw in enumerate(mne_raws):
            if raw.info["sfreq"] != info["sfreq"]:
                raise ValueError(
                    f"Cannot fill gaps if Raws have different frequencies.(0:{info['sfreq']}" + 
                    f" - {iRec}:{raw.info['sfreq']})"
                )
            if raw.ch_names != mne_raws[0].ch_names:
                raise ValueError(
                    f"Cannot fill gaps if Raws have different channel names.(0:{mne_raws[0].ch_names}" + 
                    f" - {iRec}:{raw.ch_names})"
                )
    #         if iRec > 0:
    #             if (
    #                 not mne_raws[iRec - 1].info["meas_date"]
    #                 < mne_raws[iRec].info["meas_date"]
    #             ):
    #                 raise ValueError(
    #                     "mne_raws list must be ordered regarding measurement dates."
    #                 )

        # concatenate recordings and fill gaps
        concat_raws = mne_raws[0]
        for iRec, raw in enumerate(mne_raws):
            if iRec > 0:
                # we must fill the gap between the current and the previous
                diff_in_sec = starts[iRec] - ends[iRec - 1]
                n_samples_to_add = int(diff_in_sec * info["sfreq"])

                if n_samples_to_add < 0:
                    raise ValueError(f"Overlapping raws cannot be correctly concatenated")

                if n_samples_to_add > 0:
                    gap = mne.io.RawArray(np.zeros((info["nchan"], n_samples_to_add)), info)
                    gap.annotations.append(
                        onset=0,
                        duration=n_samples_to_add / info["sfreq"],
                        description="bad_acq_skip",
                    )
                    concat_raws.append(gap)
                concat_raws.append(raw)
    else:
        concat_raws = mne_raws

    return concat_raws
    

def load_muse_csv_as_raw(filename, sfreq=256., ch_ind=[0, 1, 2, 3],
                         stim_ind=5, replace_ch_names=None, fill=False, markers=None):
    """Load CSV files into a Raw object.

    Args:
        filename (str or list): path or paths to CSV files to load

    Keyword Args:
        subject_nb (int or str): subject number. If 'all', load all
            subjects.
        session_nb (int or str): session number. If 'all', load all
            sessions.
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep
        stim_ind (int): index of the stim channel
        replace_ch_names (dict or None): dictionary containing a mapping to
            rename channels. Useful when an external electrode was used.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
    """
    n_channel = len(ch_ind)

    raw = []
    starts = []
    ends = []
    data_list = []
    for fname in filename:
        # read the file
        data = pd.read_csv(fname)
        data_list.append(data)

    data = pd.concat(data_list).sort_values("timestamps").reset_index(drop=True)
    #data = data.loc[data['timestamps'] < 40000 + data['timestamps'][0]]
    data.reset_index(drop=True,inplace=True)
    if markers is not None:
        
        data = pd.merge_asof(data, markers, on = "timestamps", 
                                 direction = "backward")
        print(data)
        data['Marker0'] = data['Marker0_y']
        data.drop(['Marker0_x','Marker0_y'], axis=1, inplace=True)
        data.dropna(axis=0, subset='Marker0', inplace=True)

    #index is timestamp
    diff = data.timestamps.diff()
    missing_idx = diff[diff > 0.1].index.tolist()
    # missing_idx = [i+1 for i in missing_idx]
    if len(missing_idx) > 0:
        missing_idx[0] = 0
    else:
        missing_idx = [0]

    missing_idx.append(len(data.index))
    print(missing_idx)
    data_sorted = [data.iloc[missing_idx[i]:missing_idx[i+1]-1] for i in range(len(missing_idx)-1) if missing_idx[i+1] - missing_idx[i] > 256]

    for data in data_sorted: #remove index
        # read the file
        data.set_index('timestamps', inplace=True)
        print(data)
        starts.append(data.index[0])
        ends.append(data.index[-1])
        
        # name of each channels
        ch_names = list(data.columns)[0:n_channel] + ['Stim']

        if replace_ch_names is not None:
            ch_names = [c if c not in replace_ch_names.keys()
                        else replace_ch_names[c] for c in ch_names]

        # type of each channels
        ch_types = ['eeg'] * n_channel + ['stim']
        if mne.__version__ <= '0.17.0':
            montage = read_montage('standard_1005') #for mne version 0.17.0 or older
        else:
            montage = make_standard_montage("standard_1005") # hxf 2/27/2021, for mne verion 0.22.0

        # get data and exclude Aux channel
        data = data.values[:, ch_ind + [stim_ind]].T

        # convert in Volts (from uVolts)
        data[:-1] *= 1e-6

        # create MNE object, refer to https://mne.tools/stable/auto_examples/io/plot_read_neo_format.html#sphx-glr-auto-examples-io-plot-read-neo-format-py
        # info = create_info(ch_names=ch_names, ch_types=ch_types,
        #                    sfreq=sfreq, montage=montage)
        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw.append(RawArray(data=data, info=info))

    # concatenate all raw objects
    if fill == True:
        raws = concatenate_and_fill_raws(raw, starts, ends)
    elif fill == False:
        raws = concatenate_raws(raw)
        
    if mne.__version__ > '0.17.0':
        raws.set_montage(montage)
    return raws


def load_data(data_dir, subject_nb=1, session_nb=1, sfreq=256.,
              ch_ind=[0, 1, 2, 3], stim_ind=5, replace_ch_names=None):
    """Load CSV files from the /data directory into a Raw object.

    Args:
        data_dir (str): directory inside /data that contains the
            CSV files to load, e.g., 'auditory/P300'

    Keyword Args:
        subject_nb (int or str): subject number. If 'all', load all
            subjects.
        session_nb (int or str): session number. If 'all', load all
            sessions.
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep
        stim_ind (int): index of the stim channel
        replace_ch_names (dict or None): dictionary containing a mapping to
            rename channels. Useful when an external electrode was used.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
    """
    if subject_nb == 'all':
        subject_nb = '*'
    if session_nb == 'all':
        session_nb = '*'

    data_path = os.path.join(
            '../', data_dir,
            'subject{}/session{}/data_*.csv'.format(subject_nb, session_nb))
    fnames = glob(data_path)

    return load_muse_csv_as_raw(fnames, sfreq=sfreq, ch_ind=ch_ind,
                                stim_ind=stim_ind,
                                replace_ch_names=replace_ch_names)

def plot_conditions(epochs, conditions=OrderedDict(), ci=97.5, n_boot=1000,
                    title='', palette=None, ylim=(-6, 6),
                    diff_waveform=(1, 2)):
    """Plot ERP conditions.

    Args:
        epochs (mne.epochs): EEG epochs

    Keyword Args:
        conditions (OrderedDict): dictionary that contains the names of the
            conditions to plot as keys, and the list of corresponding marker
            numbers as value. E.g.,

                conditions = {'Non-target': [0, 1],
                               'Target': [2, 3, 4]}

        ci (float): confidence interval in range [0, 100]
        n_boot (int): number of bootstrap samples
        title (str): title of the figure
        palette (list): color palette to use for conditions
        ylim (tuple): (ymin, ymax)
        diff_waveform (tuple or None): tuple of ints indicating which
            conditions to subtract for producing the difference waveform.
            If None, do not plot a difference waveform

    Returns:
        (matplotlib.figure.Figure): figure object
        (list of matplotlib.axes._subplots.AxesSubplot): list of axes
    """
    if isinstance(conditions, dict):
        conditions = OrderedDict(conditions)

    if palette is None:
        palette = sns.color_palette("hls", len(conditions) + 1)

    X = epochs.get_data() * 1e6
    times = epochs.times
    y = pd.Series(epochs.events[:, -1])

    fig, axes = plt.subplots(2, 2, figsize=[12, 6],
                             sharex=True, sharey=True)
    axes = [axes[1, 0], axes[0, 0], axes[0, 1], axes[1, 1]]

    for ch in range(4):
        for cond, color in zip(conditions.values(), palette):
            sns.tsplot(X[y.isin(cond), ch], time=times, color=color,
                       n_boot=n_boot, ci=ci, ax=axes[ch])

        if diff_waveform:
            diff = (np.nanmean(X[y == diff_waveform[1], ch], axis=0) -
                    np.nanmean(X[y == diff_waveform[0], ch], axis=0))
            axes[ch].plot(times, diff, color='k', lw=1)

        axes[ch].set_title(epochs.ch_names[ch])
        axes[ch].set_ylim(ylim)
        axes[ch].axvline(x=0, ymin=ylim[0], ymax=ylim[1], color='k',
                         lw=1, label='_nolegend_')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[-1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (uV)')

    if diff_waveform:
        legend = (['{} - {}'.format(diff_waveform[1], diff_waveform[0])] +
                  list(conditions.keys()))
    else:
        legend = conditions.keys()
    axes[-1].legend(legend)
    sns.despine()
    plt.tight_layout()

    if title:
        fig.suptitle(title, fontsize=20)

    return fig, axes


def plot_highlight_regions(x, y, hue, hue_thresh=0, xlabel='', ylabel='',
                           legend_str=()):
    """Plot a line with highlighted regions based on additional value.

    Plot a line and highlight ranges of x for which an additional value
    is lower than a threshold. For example, the additional value might be
    pvalues, and the threshold might be 0.05.

    Args:
        x (array_like): x coordinates
        y (array_like): y values of same shape as `x`

    Keyword Args:
        hue (array_like): values to be plotted as hue based on `hue_thresh`.
            Must be of the same shape as `x` and `y`.
        hue_thresh (float): threshold to be applied to `hue`. Regions for which
            `hue` is lower than `hue_thresh` will be highlighted.
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        legend_str (tuple): legend for the line and the highlighted regions

    Returns:
        (matplotlib.figure.Figure): figure object
        (list of matplotlib.axes._subplots.AxesSubplot): list of axes
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)

    axes.plot(x, y, lw=2, c='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    kk = 0
    a = []
    while kk < len(hue):
        if hue[kk] < hue_thresh:
            b = kk
            kk += 1
            while kk < len(hue):
                if hue[kk] > hue_thresh:
                    break
                else:
                    kk += 1
            a.append([b, kk - 1])
        else:
            kk += 1

    st = (x[1] - x[0]) / 2.0
    for p in a:
        axes.axvspan(x[p[0]]-st, x[p[1]]+st, facecolor='g', alpha=0.5)
    plt.legend(legend_str)
    sns.despine()

    return fig, axes
