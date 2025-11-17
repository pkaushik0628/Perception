import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, decimate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Utility functions
# -----------------------------
def bandpass_filter(data, lowcut=1, highcut=100, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def compute_attention(eye_x, eye_y, x_max=1920, y_max=1080):
    attention = ((eye_x >= 0) & (eye_x <= x_max) & (eye_y >= 0) & (eye_y <= y_max)).astype(int)
    attention = pd.Series(attention).rolling(20, center=True).mean().fillna(0)
    return (attention > 0.5).astype(int)

def find_segments(attention, min_length=50):
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
def extract_region_signals(nwb_file_path, electrode_type='micro', regions=None, top_n=8):
    io = NWBHDF5IO(nwb_file_path, 'r')
    nwbfile = io.read()

    electrodes_df = nwbfile.electrodes.to_dataframe()
    if regions is None:
        regions = electrodes_df['location'].dropna().unique()

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
                    sig_data = bandpass_filter(sig_data, lowcut=1, highcut=100, fs=1000)
                    signals[f"{region}_{idx}"] = sig_data
                    break
        if len(signals) > top_n:
            power_sorted = sorted(signals.items(), key=lambda x: np.sum(x[1] ** 2), reverse=True)
            signals = dict(power_sorted[:top_n])
        region_signals[region] = signals

    io.close()
    return region_signals

# -----------------------------
# Prepare sequence dataset
# -----------------------------
def prepare_sequence_dataset(participant_ids, regions, window_size=500, step_size=250, top_n=8):
    all_X, all_y = [], []
    electrode_sets = []

    # Determine common electrodes across participants
    for pid in participant_ids:
        nwb_path = f"/Users/padmanabh/PycharmProjects/EEG&fMRI/000623/sub-CS{pid}/sub-CS{pid}_ses-P{pid}CSR1_behavior+ecephys.nwb"
        region_signals = extract_region_signals(nwb_path, electrode_type='micro', regions=regions, top_n=top_n)
        electrode_names = []
        for sig_dict in region_signals.values():
            electrode_names.extend(sig_dict.keys())
        electrode_sets.append(set(electrode_names))

    common_electrodes = set.intersection(*electrode_sets)
    if len(common_electrodes) == 0:
        raise ValueError("No common electrodes found! Reduce top_n or check regions.")
    print(f"Using {len(common_electrodes)} common electrodes.")

    # Extract windows and labels
    for pid in participant_ids:
        nwb_path = f"/Users/padmanabh/PycharmProjects/EEG&fMRI/000623/sub-CS{pid}/sub-CS{pid}_ses-P{pid}CSR1_behavior+ecephys.nwb"
        io = NWBHDF5IO(nwb_path, 'r')
        nwbfile = io.read()

        # Eye-tracking attention segments
        eye_tracking = nwbfile.processing['behavior'].data_interfaces['EyeTracking']
        eye_first_series = list(eye_tracking.spatial_series.values())[0]
        eye_data = np.array(eye_first_series.data[:])
        eye_x, eye_y = eye_data[:, 0], eye_data[:, 1]
        attention = compute_attention(eye_x, eye_y)
        segments = find_segments(attention, min_length=50)

        # Electrode signals
        region_signals = extract_region_signals(nwb_path, electrode_type='micro', regions=regions, top_n=top_n)
        signals_selected = {}
        for sig_dict in region_signals.values():
            for name, sig in sig_dict.items():
                if name in common_electrodes:
                    factor = max(1, len(sig) // len(eye_x))
                    sig_ds = decimate(sig, factor)
                    signals_selected[name] = sig_ds

        for start_idx, end_idx, label in segments:
            for ws_start in range(start_idx, end_idx - window_size + 1, step_size):
                window = [signals_selected[e][ws_start:ws_start + window_size] for e in sorted(common_electrodes)]
                window = np.stack(window, axis=1)
                all_X.append(window)
                all_y.append(label)

        io.close()

    X = np.stack(all_X, axis=0)
    y = np.array(all_y)
    print(f"Dataset shape: {X.shape}, Labels distribution: {np.unique(y, return_counts=True)}")
    return X, y, sorted(common_electrodes)

# -------------------------------
# Models
# -----------------------------
def build_transformer_model(input_shape, num_heads=4, ff_dim=64):
    inputs = layers.Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(ff_dim, activation='relu', name='transformer_embedding')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='transformer_logits')(x)
    return models.Model(inputs, outputs)

def build_helper_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(input_shape[1], 1, padding='same')(x)
    return models.Model(inputs, x)

# ----------------------------------
# Focal loss
# -----------------------------
def focal_loss(gamma=2., alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = alpha * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss_fn

# -----------------------------
# Main pipeline
# --------------------------------
participant_ids = [41,42,43,44,47,48,49,51,53,54,55]
regions_of_interest = [
    'LSPE','Left ACC','Left amygdala','Left hippocampus','Left preSMA','Left vmPFC',
    'RSPE','Right ACC','Right amygdala','Right hippocampus','Right preSMA','Right vmPFC'
]

# Prepare dataset
X, y, common_electrodes = prepare_sequence_dataset(participant_ids, regions_of_interest, window_size=500, step_size=250, top_n=8)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
num_timesteps, num_electrodes = X_train.shape[1], X_train.shape[2]

# -----------------------------
# Scale per electrode
# -------------------------------
scalers = []
X_train_scaled = np.zeros_like(X_train)
X_test_scaled = np.zeros_like(X_test)
for i in range(num_electrodes):
    scaler = StandardScaler()
    X_train_scaled[:,:,i] = scaler.fit_transform(X_train[:,:,i])
    X_test_scaled[:,:,i] = scaler.transform(X_test[:,:,i])
    scalers.append(scaler)

# --------------------------------
# Build and train transformer baseline
# -----------------------------
transformer = build_transformer_model((num_timesteps, num_electrodes))
transformer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
transformer.fit(
    X_train_scaled, y_train,
    validation_split=0.2, epochs=50, batch_size=16,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
transformer.trainable = False  # freeze transformer

# -----------------------------
# Build helper model
# ---------------------------------
helper = build_helper_model((num_timesteps, num_electrodes))
helper.compile(optimizer='adam', loss=focal_loss())
helper.summary()

# ----------------------------------
# Train helper to mimic transformer input
# ------------------------------------
# Feed helper output into frozen transformer
inp = layers.Input(shape=(num_timesteps, num_electrodes))
transformed = helper(inp)
logits = transformer(transformed)
helper_pipeline = models.Model(inp, logits)
helper_pipeline.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

helper_pipeline.fit(
    X_train_scaled, y_train,
    validation_split=0.2, epochs=50, batch_size=16,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# -----------------------------
# Helper function for test data prediction
# -----------------------------
def predict_with_helper(helper_model, transformer_model, X_raw, scalers, threshold=0.5):
    """
    X_raw: shape (num_samples, timesteps, electrodes)
    scalers: list of fitted StandardScaler per electrode
    """
    X_scaled = np.zeros_like(X_raw)
    for i, scaler in enumerate(scalers):
        X_scaled[:,:,i] = scaler.transform(X_raw[:,:,i])
    X_transformed = helper_model.predict(X_scaled)
    y_prob = transformer_model.predict(X_transformed)
    y_pred = (y_prob.flatten() >= threshold).astype(int)
    return y_prob.flatten(), y_pred

# ---------------------------------
# Evaluate on test set
# --------------------------------------
y_prob, y_pred = predict_with_helper(helper, transformer, X_test, scalers)

print(classification_report(y_test, y_pred, digits=3))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Helper + Transformer Predictions")
plt.show()
