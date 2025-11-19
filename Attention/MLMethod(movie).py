import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.signal import welch, butter, filtfilt, decimate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.ensemble import RUSBoostClassifier
import re

# -----------------------------
# Utility functions
# -----------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, data, axis=0)

def get_stimulus_screen_limits(nwb_path):
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()

        # Access the EyeTracking object
        eye_tracking_obj = nwbfile.processing['behavior']['EyeTracking']

        # Iterate over SpatialSeries in case there are multiple
        for ss in eye_tracking_obj.spatial_series.values():
            comments = ss.comments

            # Extract display_area from comments
            area_match = re.search(r'display_area=([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+)', comments)
            if area_match:
                x_start = float(area_match.group(1))
                y_start = float(area_match.group(2))
                x_end = float(area_match.group(3))
                y_end = float(area_match.group(4))

                top_left = (x_start, y_start)
                bottom_right = (x_end, y_end)
                return top_left, bottom_right

        raise ValueError("display_area not found in any SpatialSeries comments")


def compute_attention(eye_x, eye_y, nwb_path):
    top_left, bottom_right = get_stimulus_screen_limits(nwb_path)
    x_min, y_min = top_left
    x_max, y_max = bottom_right
    attention = ((eye_x >= x_min) & (eye_x <= x_max) & (eye_y >= y_min) & (eye_y <= y_max)).astype(int)
    attention = pd.Series(attention).rolling(20, center=True).mean().fillna(0)
    return (attention > 0.5).astype(int)

def find_segments(attention, min_length=10):
    segments = []
    current_label = attention[0]
    start_idx = 0
    for i in range(1, len(attention)):
        if attention[i] != current_label:
            if i - start_idx >= min_length:
                segments.append((start_idx, i, current_label))
            start_idx = i
            current_label = attention[i]
    if len(attention) - start_idx >= min_length:
        segments.append((start_idx, len(attention), current_label))
    return segments

# -----------------------------
# NWB signal extraction
# -----------------------------
def extract_region_signals(nwb_file_path, electrode_type='micro', regions=None, top_n=8, fs_lfp=1000, fs_eye=500):
    io = NWBHDF5IO(nwb_file_path, 'r')
    nwbfile = io.read()

    electrodes_df = nwbfile.electrodes.to_dataframe()
    if regions is None:
        regions = electrodes_df['location'].dropna().unique()

    # Match group_name containing "macro" or "macros"
    electrodes_df = electrodes_df[electrodes_df['group_name'].str.contains('micro', case=False, na=False)]
    data_interface = nwbfile.processing['ecephys'].data_interfaces.get(f'LFP_{electrode_type}', None)
    if data_interface is None:
        io.close()
        return {}

    region_signals = {}
    for region in regions:
        region_df = electrodes_df[electrodes_df['location'].str.contains(region, case=False, na=False)]
        signals = {}
        for idx in region_df.index:
            for es in data_interface.electrical_series.values():
                electrode_table = es.electrodes.to_dataframe()
                if idx in electrode_table.index:
                    col_idx = electrode_table.index.get_loc(idx)
                    sig_data = np.array(es.data[:, col_idx])
                    signals[f"{region}_{idx}"] = sig_data
                    break
        # Pick top-N electrodes by total power
        if len(signals) > top_n:
            power_sorted = sorted(signals.items(), key=lambda x: np.sum(x[1]**2), reverse=True)
            signals = dict(power_sorted[:top_n])
        region_signals[region] = signals

    io.close()
    return region_signals

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(lfp_data, segments, fs):
    features, labels = [], []
    for start, end, label in segments:
        feat_row = []
        for electrode_name, sig in lfp_data.items():
            segment = sig[start:end]
            mean_val = np.mean(segment)
            rms_val = np.sqrt(np.mean(segment**2))
            freqs, psd = welch(segment, fs=fs, nperseg=min(len(segment), fs*2))
            total_power = np.sum(psd)
            feat_row.extend([mean_val, rms_val, total_power])
        features.append(feat_row)
        labels.append(label)
    return np.array(features), np.array(labels)

# -----------------------------
# Prepare dataset for ML
# -----------------------------
def prepare_dataset(participant_ids, regions, fs_lfp=1000, fs_eye=500):
    all_X, all_y = [], []
    for pid in participant_ids:
        nwb_path = f"/Users/padmanabh/PycharmProjects/EEG&fMRI/000623/sub-CS{pid}/sub-CS{pid}_ses-P{pid}CSR1_behavior+ecephys.nwb"
        io = NWBHDF5IO(nwb_path, "r")
        nwbfile = io.read()

        # Eye-tracking
        eye_tracking = nwbfile.processing['behavior'].data_interfaces['EyeTracking']
        eye_first_series = list(eye_tracking.spatial_series.values())[0]
        eye_data = np.array(eye_first_series.data[:])
        eye_x, eye_y = eye_data[:,0], eye_data[:,1]
        attention = compute_attention(eye_x, eye_y, nwb_path)
        segments = find_segments(attention, min_length=50)

        # LFP signals
        region_signals = extract_region_signals(nwb_file_path=nwb_path, electrode_type='micro', regions=regions, fs_lfp=fs_lfp, fs_eye=fs_eye)

        # Downsample to eye-tracking fs
        lfp_signals = {}
        for sig_dict in region_signals.values():
            for name, sig in sig_dict.items():
                factor = max(1, len(sig)//len(eye_x))
                lfp_signals[name] = decimate(sig, factor)

        X_part, y_part = extract_features(lfp_signals, segments, fs=fs_lfp)
        all_X.append(X_part)
        all_y.append(y_part)
        io.close()

    X = np.vstack(all_X)
    y = np.hstack(all_y)
    return X, y

# -----------------------------
# Run ML
# -----------------------------
participant_ids = [41,
                   42,
                   43,
                   44,
                   47,48,49,51,53,54,55,
                   57, 58, 60
                   ]

regions_of_interest = [
    #'LSPE',
    'Left ACC','Left amygdala','Left hippocampus','Left preSMA',
    #'Left vmPFC',
    #'RSPE',
    'Right ACC','Right amygdala','Right hippocampus','Right preSMA',
    #'Right vmPFC'
]

X, y = prepare_dataset(participant_ids, regions_of_interest)

print(f"Dataset shape: {X.shape}, Labels distribution: {np.unique(y, return_counts=True)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classifier
#clf = RandomForestClassifier(n_estimators=200, random_state=42)

clf = RUSBoostClassifier(
    n_estimators=100,
    random_state=42
)

clf.fit(X_train_scaled, y_train)
# Predictions & Evaluation
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred, digits=3))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

##SANITY CHECK
#Calculate the majority ratio
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))
majority_class = unique[np.argmax(counts)]
majority_ratio = np.max(counts) / np.sum(counts)

print("\n=== CLASS DISTRIBUTION ===")
for cls, cnt in class_counts.items():
    print(f"Class {cls}: {cnt} samples")
print(f"Majority class: {majority_class}")
print(f"Majority class ratio (expected chance baseline): {majority_ratio:.4f}")

#Revisit the test dataset
y_test_shuffled = np.random.permutation(y_test)
print("Label Distribution after shuffling: ", np.unique(y_test_shuffled, return_counts=True))
#Predict probabilities using the same trained model
preds_scrambled = clf.predict(X_test_scaled)

from sklearn.metrics import f1_score, accuracy_score

acc_scrambled = accuracy_score(y_test_shuffled, preds_scrambled)

if acc_scrambled > majority_ratio + 0.05:
    print("Shuffled label performance is higher than expected")
else:
    print("Sanity Check passed")
    print("Scrambled Accuracy:", acc_scrambled)
