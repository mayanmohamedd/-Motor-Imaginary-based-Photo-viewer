#take a sample file from data to understand it

import mne

raw=mne.io.read_raw_gdf('archive (3)\A04E.gdf',
                         eog=['EOG-left', 'EOG-central', 'EOG-right'])
# Get the number of channels before dropping
num_channels_before = raw.info['nchan']
print("Number of channels before dropping:", num_channels_before)

raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

# Get the number of channels before dropping
num_channels_before = raw.info['nchan']
print("Number of channels before dropping:", num_channels_before)

print("Channel names after dropping:", raw.ch_names)

events=mne.events_from_annotations(raw)
print("event[0]",events[0])
print("event[1]",events[1])


"""
1023  Rejected trial           event1
1072  Eye movements            event2
276   (eyes open)              event3
277   (eyes closed)            event4
32766  Start of a new run      event5
768  Start of a trial          event6
769   left (class 1)           event7
770   right (class 2)          event8
771  foot (class 3)            event9
772   tongue (class 4)         event10
"""


event_dict={
 'reject':1,
 'eye move':2,
 'eye open':3,
 'eye close':4,
 'new run':5,
 'new trial':6,
 'class 1':7,
 'class 2':8,
 'class 3':9,
 'class 4':10,

}


fig = mne.viz.plot_events(events[0], event_id=event_dict, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)

epochs = mne.Epochs(raw, events[0], event_id=[7,8],tmin= -0.1, tmax=0.7, preload=True)
# Get the number of epochs
n_epochs = len(epochs)
print(f"Number of epochs: {n_epochs}")
print(epochs.get_data().shape)

#to know whcih column is class column
label=epochs.events
print("label of class is",label)


label=epochs.events[:,-1]
print(len(label))


evoked_0 = epochs['7'].average()
evoked_1 = epochs['8'].average()


#plotting classes
dicts={'class0':evoked_0,'class1':evoked_1}
mne.viz.plot_compare_evokeds(dicts)



