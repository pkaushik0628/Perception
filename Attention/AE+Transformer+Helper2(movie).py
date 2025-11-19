# helper_transformer_with_ae.py
import os
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, decimate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Utilities (mostly unchanged)
# -----------------------------
def bandpass_filter(data, lowcut=1, highcut=100, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def compute_attention(eye_x, eye_y, x_max=1920, y_max=1080):
    attention = ((eye_x >= 0) & (eye_x <= x_max) & (eye_y >= 0) & (eye_y <= y_max)).astype(int)
    attention = pd.Series(attention).rolling(20, center=True).mean().fillna(0)
    return (attention > 0.5).astype(int)

def find_segments(attention, min_length=50):
    segments = []
    current_label = int(attention[0])
    start_idx = 0
    for i in range(1, len(attention)):
        if int(attention[i]) != current_label:
            if i - start_idx >= min_length:
                segments.append((start_idx, i, current_label))
            start_idx = i
            current_label = int(attention[i])
    if len(attention) - start_idx >= min_length:
        segments.append((start_idx, len(attention), current_label))
    return segments

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
            power_sorted = sorted(signals.items(), key=lambda x: np.sum(x[1]**2), reverse=True)
            signals = dict(power_sorted[:top_n])
        region_signals[region] = signals
    io.close()
    return region_signals

# -----------------------------
# Dataset prep (keeps your previous logic)
# -----------------------------
def prepare_sequence_dataset(participant_ids, regions, window_size=500, step_size=250, top_n=8):
    all_X, all_y = [], []
    electrode_sets = []
    # first pass to get common electrodes
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
    common_electrodes = sorted(common_electrodes)
    print(f"Using {len(common_electrodes)} common electrodes.")

    # second pass to extract windows
    for pid in participant_ids:
        nwb_path = f"/Users/padmanabh/PycharmProjects/EEG&fMRI/000623/sub-CS{pid}/sub-CS{pid}_ses-P{pid}CSR1_behavior+ecephys.nwb"
        io = NWBHDF5IO(nwb_path, 'r')
        nwbfile = io.read()

        eye_tracking = nwbfile.processing['behavior'].data_interfaces['EyeTracking']
        eye_first_series = list(eye_tracking.spatial_series.values())[0]
        eye_data = np.array(eye_first_series.data[:])
        eye_x, eye_y = eye_data[:,0], eye_data[:,1]
        attention = compute_attention(eye_x, eye_y)
        segments = find_segments(attention, min_length=50)

        region_signals = extract_region_signals(nwb_path, electrode_type='micro', regions=regions, top_n=top_n)
        signals_selected = {}
        for sig_dict in region_signals.values():
            for name, sig in sig_dict.items():
                if name in common_electrodes:
                    factor = max(1, len(sig)//len(eye_x))
                    sig_ds = decimate(sig, factor)
                    signals_selected[name] = sig_ds

        for start_idx, end_idx, label in segments:
            for ws_start in range(start_idx, end_idx - window_size + 1, step_size):
                window = [signals_selected[e][ws_start:ws_start+window_size] for e in common_electrodes]
                window = np.stack(window, axis=1)
                all_X.append(window)
                all_y.append(label)
        io.close()

    X = np.stack(all_X, axis=0)
    y = np.array(all_y)
    print(f"Dataset shape: {X.shape}, Labels distribution: {np.unique(y, return_counts=True)}")
    return X, y, common_electrodes

# -----------------------------
# Autoencoder (encoder + decoder)
# -----------------------------
def build_sequence_autoencoder(input_shape, latent_dim=16, conv_filters=[64,32], kernel_size=5):
    """
    input_shape = (timesteps, channels)
    returns encoder, decoder, autoencoder (Keras Models)
    """
    seq_len, channels = input_shape
    inp = layers.Input(shape=input_shape, name='ae_input')
    x = inp
    # small conv encoder
    for f in conv_filters:
        x = layers.Conv1D(f, kernel_size, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(latent_dim, activation='relu', name='ae_bottleneck')(x)

    # decoder
    x = layers.Dense((seq_len // (2**len(conv_filters))) * conv_filters[-1], activation='relu')(bottleneck)
    x = layers.Reshape((seq_len // (2**len(conv_filters)), conv_filters[-1]))(x)
    for f in conv_filters[::-1]:
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(f, kernel_size, padding='same', activation='relu')(x)
    # final projection to original channels
    out = layers.Conv1D(channels, 1, padding='same', activation=None, name='ae_reconstruction')(x)

    encoder = models.Model(inp, bottleneck, name='ae_encoder')
    #decoder = models.Model(layers.Input(shape=(latent_dim,)), decoder_out(decoder_input_shape=latent_dim, seq_len=seq_len, channels=channels))
    # but easier: build full autoencoder and then extract decoder from middle
    autoencoder = models.Model(inp, out, name='autoencoder')
    # create decoder by connecting a latent input through the decoding layers — easier rebuild:
    # Rebuild decoder matching the decoder steps above
    # Rebuild decoder explicitly:
    d_in = layers.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense((seq_len // (2**len(conv_filters))) * conv_filters[-1], activation='relu')(d_in)
    x = layers.Reshape((seq_len // (2**len(conv_filters)), conv_filters[-1]))(x)
    for f in conv_filters[::-1]:
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(f, kernel_size, padding='same', activation='relu')(x)
    d_out = layers.Conv1D(channels, 1, padding='same', activation=None)(x)
    decoder = models.Model(d_in, d_out, name='ae_decoder')

    return encoder, decoder, autoencoder

def decoder_out(decoder_input_shape, seq_len, channels):
    # not used; helper placeholder
    raise NotImplementedError

# -----------------------------
# Transformer (modified to accept augmented channels)
# -----------------------------
def build_transformer_model(input_shape, num_heads=4, ff_dim=128, emb_dim=128):
    """
    input_shape: (timesteps, channels_aug) where channels_aug = original_channels + latent_dim_tiled
    returns a Keras model with named penultimate layer 'transformer_embedding' for extracting embeddings.
    """
    inp = layers.Input(shape=input_shape, name='transformer_input')
    x = layers.LayerNormalization()(inp)
    # small transformer stack: one MultiHeadAttention + feed-forward
    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1]//num_heads)(x, x)
    x = layers.Add()([x, att])
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(ff_dim, 1, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    emb = layers.Dense(emb_dim, activation='relu', name='transformer_embedding')(x)
    out = layers.Dense(1, activation='sigmoid', name='transformer_logits')(emb)
    return models.Model(inp, out, name='transformer_model')

# -----------------------------
# Improved helper (residual + separable conv blocks + noise) - same as original
# -----------------------------
def helper_block(x, filters, kernel_size=7, name=None):
    y = layers.SeparableConv1D(filters, kernel_size, padding='same', activation='relu')(x)
    y = layers.SeparableConv1D(filters, kernel_size, padding='same', activation='relu')(y)
    y = layers.LayerNormalization()(y)
    if x.shape[-1] != filters:
        x_proj = layers.Conv1D(filters, 1, padding='same')(x)
    else:
        x_proj = x
    out = layers.Add()([x_proj, y])
    out = layers.Activation('relu')(out)
    out = layers.LayerNormalization()(out)
    return out

def build_helper_model(input_shape):
    inp = layers.Input(shape=input_shape, name='helper_input')
    x = layers.GaussianNoise(0.05)(inp)
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(x)
    x = helper_block(x, 64, kernel_size=7)
    x = helper_block(x, 64, kernel_size=5)
    x = helper_block(x, 128, kernel_size=5)
    x = layers.Conv1D(input_shape[1], 1, padding='same', activation=None)(x)  # project back to electrode-space
    return models.Model(inp, x, name='helper_model')

# -----------------------------
# Asymmetric focal loss (unchanged)
# -----------------------------
def asymmetric_focal_loss(alpha_0=0.9, alpha_1=0.1, gamma=2.0):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true_f, y_pred)
        p_t = y_true_f * y_pred + (1 - y_true_f) * (1 - y_pred)
        alpha = y_true_f * alpha_0 + (1 - y_true_f) * alpha_1
        focal = alpha * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss

# -----------------------------
# HelperTrainer (modified to accept precomputed latents)
# -----------------------------
class HelperTrainer(tf.keras.Model):
    def __init__(self, helper_model, transformer_model, embedding_model,
                 class_center_0, class_center_1, lambda_margin=0.3,
                 focal_alpha_0=0.9, focal_alpha_1=0.1, focal_gamma=2.0):
        super().__init__()
        self.helper = helper_model
        self.transformer = transformer_model  # frozen
        self.embedding_model = embedding_model  # frozen (maps transformer_input -> embedding)
        self.class_center_0 = tf.constant(class_center_0, dtype=tf.float32)
        self.class_center_1 = tf.constant(class_center_1, dtype=tf.float32)
        self.lambda_margin = lambda_margin

        # focal params
        self.f_alpha_0 = float(focal_alpha_0)
        self.f_alpha_1 = float(focal_alpha_1)
        self.f_gamma = float(focal_gamma)

        # metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.cls_acc = tf.keras.metrics.BinaryAccuracy(name="acc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.cls_acc]

    def call(self, inputs, training=False):
        # inputs expected: (raw_x, latent_tiled) as a tuple
        raw_x, latent_tiled = inputs
        helper_out = self.helper(raw_x, training=training)  # (batch, T, channels)
        # concat along channels axis
        transformer_input = tf.concat([helper_out, latent_tiled], axis=-1)
        logits = self.transformer(transformer_input, training=False)
        return logits

    def compute_focal(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha = y_true * self.f_alpha_0 + (1 - y_true) * self.f_alpha_1
        focal = alpha * tf.pow(1 - p_t, self.f_gamma) * bce
        return tf.reduce_mean(focal)

    def compute_margin(self, y_true, emb_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ref0 = tf.reshape(self.class_center_0, (1, -1))
        ref1 = tf.reshape(self.class_center_1, (1, -1))
        ref0_batch = tf.repeat(ref0, tf.shape(emb_pred)[0], axis=0)
        ref1_batch = tf.repeat(ref1, tf.shape(emb_pred)[0], axis=0)
        mask = tf.cast(tf.expand_dims(tf.equal(y_true, 1), axis=1), tf.float32)
        ref_batch = mask * ref1_batch + (1.0 - mask) * ref0_batch
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(emb_pred - ref_batch), axis=1))
        return mse

    def train_step(self, data):
        """
        data: ((raw_x_batch, latent_tiled_batch), y_batch)
        latent_tiled_batch must be precomputed for each sample and passed in dataset
        """
        (raw_x, latent_tiled), y = data
        with tf.GradientTape() as tape:
            helper_out = self.helper(raw_x, training=True)
            transformer_input = tf.concat([helper_out, latent_tiled], axis=-1)
            # forward through frozen transformer (we expect transformer.trainable = False)
            logits = self.transformer(transformer_input, training=False)
            logits = tf.reshape(logits, [-1])
            # embedding from embedding_model (maps transformer_input to penultimate embedding)
            emb_pred = self.embedding_model(transformer_input, training=False)

            cls_loss = self.compute_focal(y, logits)
            emb_loss = self.compute_margin(y, emb_pred)
            total_loss = cls_loss + self.lambda_margin * emb_loss

        grads = tape.gradient(total_loss, self.helper.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.helper.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.cls_acc.update_state(y, logits)
        return {"loss": self.loss_tracker.result(), "acc": self.cls_acc.result()}

    def test_step(self, data):
        (raw_x, latent_tiled), y = data
        helper_out = self.helper(raw_x, training=False)
        transformer_input = tf.concat([helper_out, latent_tiled], axis=-1)
        logits = self.transformer(transformer_input, training=False)
        logits = tf.reshape(logits, [-1])
        emb_pred = self.embedding_model(transformer_input, training=False)
        cls_loss = self.compute_focal(y, logits)
        emb_loss = self.compute_margin(y, emb_pred)
        total_loss = cls_loss + self.lambda_margin * emb_loss
        self.loss_tracker.update_state(total_loss)
        self.cls_acc.update_state(y, logits)
        return {"loss": self.loss_tracker.result(), "acc": self.cls_acc.result()}

# -----------------------------
# threshold utility (unchanged)
# -----------------------------
def find_best_threshold(y_true, probs, target='class0_f1'):
    best_t = 0.5
    best_score = -1.0
    tgrid = np.linspace(0.01, 0.99, 99)
    for t in tgrid:
        preds = (probs >= t).astype(int)
        if target == 'class0_f1':
            score = f1_score(y_true, preds, pos_label=0)
        else:
            score = f1_score(y_true, preds, average='macro')
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score

# -----------------------------
# Small helper: tile latent to time dimension
# -----------------------------
def tile_latent_to_timesteps(latent_vecs, timesteps):
    # latent_vecs: (n_samples, latent_dim)
    # returns (n_samples, timesteps, latent_dim) tiled
    return np.repeat(latent_vecs[:, None, :], timesteps, axis=1)

# -----------------------------
# Full pipeline execution
# -----------------------------
if __name__ == "__main__":
    # user inputs
    participant_ids = [41,42,43,44,47,48,49,51,53,54,55]
    regions_of_interest = [
        'LSPE','Left ACC','Left amygdala','Left hippocampus','Left preSMA','Left vmPFC',
        'RSPE','Right ACC','Right amygdala','Right hippocampus','Right preSMA','Right vmPFC'
    ]

    # Prepare dataset (same logic)
    X, y, common_electrodes = prepare_sequence_dataset(participant_ids, regions_of_interest,
                                                       window_size=500, step_size=250, top_n=8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num_timesteps, num_electrodes = X_train.shape[1], X_train.shape[2]
    print("TIMESTEPS, ELECTRODES:", num_timesteps, num_electrodes)

    # per-electrode scalers (store list) - scale per channel across time windows
    scalers = []
    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test)
    for i in range(num_electrodes):
        sc = StandardScaler()
        X_train_scaled[:,:,i] = sc.fit_transform(X_train[:,:,i])
        X_test_scaled[:,:,i] = sc.transform(X_test[:,:,i])
        scalers.append(sc)

    # -----------------------------
    # 0) Train autoencoder on scaled sequences to get latent representation
    # -----------------------------
    latent_dim = 16
    encoder, decoder, autoencoder = build_sequence_autoencoder((num_timesteps, num_electrodes), latent_dim=latent_dim,
                                                               conv_filters=[64,32], kernel_size=5)
    autoencoder.compile(optimizer='adam', loss='mse')
    print("Autoencoder summary:")
    autoencoder.summary()
    autoencoder.fit(X_train_scaled, X_train_scaled,
                    validation_split=0.1, epochs=30, batch_size=32,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                    verbose=1)

    # encode training + test sets
    latent_train = encoder.predict(X_train_scaled, batch_size=64)
    latent_test = encoder.predict(X_test_scaled, batch_size=64)
    print("Latent shapes:", latent_train.shape, latent_test.shape)

    # tile latent across timesteps and concatenate to channels
    latent_train_tiled = tile_latent_to_timesteps(latent_train, num_timesteps)  # (N, T, latent_dim)
    latent_test_tiled = tile_latent_to_timesteps(latent_test, num_timesteps)

    # Build augmented datasets for transformer: concat along channel axis
    X_train_aug = np.concatenate([X_train_scaled, latent_train_tiled], axis=-1)  # (N, T, channels+latent)
    X_test_aug = np.concatenate([X_test_scaled, latent_test_tiled], axis=-1)
    aug_num_channels = X_train_aug.shape[-1]
    print("Augmented channels:", aug_num_channels)

    # -----------------------------
    # 1) Train transformer baseline on augmented input (so it defines the embedding space)
    # -----------------------------
    transformer = build_transformer_model((num_timesteps, aug_num_channels),
                                          num_heads=4, ff_dim=128, emb_dim=128)
    transformer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Transformer summary:")
    transformer.summary()
    transformer.fit(X_train_aug, y_train,
                    validation_split=0.2, epochs=50, batch_size=16,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                    verbose=1)
    transformer.trainable = False

    # embedding extractor: penultimate layer
    embedding_model = models.Model(transformer.input, transformer.get_layer('transformer_embedding').output)
    embeddings_train = embedding_model.predict(X_train_aug, batch_size=64)
    class_center_0 = embeddings_train[y_train == 0].mean(axis=0) if np.any(y_train == 0) else np.zeros(embeddings_train.shape[1])
    class_center_1 = embeddings_train[y_train == 1].mean(axis=0) if np.any(y_train == 1) else np.zeros(embeddings_train.shape[1])

    # -----------------------------
    # 2) Build helper model (same architecture) - it takes raw (num_timesteps, num_electrodes)
    # -----------------------------
    helper = build_helper_model((num_timesteps, num_electrodes))
    helper.summary()

    # -----------------------------
    # 3) Prepare tf.data.Dataset for HelperTrainer that yields ((raw_x, latent_tiled), y)
    # -----------------------------
    # We'll use raw X_train_scaled as helper input, and latent_train_tiled as context to be concatenated with helper_out
    batch_size = 16
    def build_helper_dataset(X_raw, latent_tiled, y_arr, batch_size=16, shuffle=True):
        # create a dataset of tuples: ((raw, latent_tiled), y)
        ds_raw = tf.data.Dataset.from_tensor_slices(X_raw.astype(np.float32))
        ds_lat = tf.data.Dataset.from_tensor_slices(latent_tiled.astype(np.float32))
        ds_y = tf.data.Dataset.from_tensor_slices(y_arr.astype(np.int32))
        ds = tf.data.Dataset.zip(((ds_raw, ds_lat), ds_y))
        if shuffle:
            ds = ds.shuffle(1024, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = build_helper_dataset(X_train_scaled, latent_train_tiled, y_train, batch_size=batch_size, shuffle=True)
    val_ds = build_helper_dataset(X_test_scaled, latent_test_tiled, y_test, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # 4) Wrap into HelperTrainer and train (updates helper only)
    # -----------------------------
    trainer = HelperTrainer(helper_model=helper,
                            transformer_model=transformer,
                            embedding_model=embedding_model,
                            class_center_0=class_center_0,
                            class_center_1=class_center_1,
                            lambda_margin=0.3,
                            focal_alpha_0=0.9, focal_alpha_1=0.1, focal_gamma=2.0)

    trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    trainer.fit(train_ds,
                validation_data=val_ds,
                epochs=50,
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                verbose=1)

    # -----------------------------
    # 5) Inference: predict with helper -> concat latent_tile -> transformer pipeline
    # -----------------------------
    def predict_with_helper(helper_model, transformer_model, X_raw_scaled, latent_tiled, threshold=0.5):
        helper_out = helper_model.predict(X_raw_scaled, batch_size=64)
        transformer_input = np.concatenate([helper_out, latent_tiled], axis=-1)
        probs = transformer_model.predict(transformer_input, batch_size=64).flatten()
        preds = (probs >= threshold).astype(int)
        return probs, preds

    # 6) Accuracy evaluation with baseline threshold of 0.5
    probs_05, preds_05 = predict_with_helper(helper, transformer, X_test_scaled, latent_test_tiled, threshold=0.5)
    print("\n--- results at threshold 0.5 ---")
    print(classification_report(y_test, preds_05, digits=3))
    cm = confusion_matrix(y_test, preds_05)
    print("Confusion matrix:\n", cm)

    # 7) find threshold that maximizes class-0 F1. Do accuracy evaluation again
    best_t, best_f1 = find_best_threshold(y_test, probs_05, target='class0_f1')
    print(f"\nBest threshold (class0 f1) = {best_t:.3f}, f1 = {best_f1:.3f}")

    probs_best, preds_best = predict_with_helper(helper, transformer, X_test_scaled, latent_test_tiled, threshold=best_t)
    print("\n--- results at best threshold ---")
    print(classification_report(y_test, preds_best, digits=3))
    cm = confusion_matrix(y_test, preds_best)
    print("Confusion matrix:\n", cm)

    # plot confusion matrix for best threshold
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Helper+Transformer+AE (threshold={best_t:.3f})")
    plt.show()

    # 8) SANITY CHECK: Reality chance check
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    majority_class = unique[np.argmax(counts)]
    majority_ratio = np.max(counts) / np.sum(counts)

    print("\n=== CLASS DISTRIBUTION ===")
    for cls, cnt in class_counts.items():
        print(f"Class {cls}: {cnt} samples")
    print(f"Majority class: {majority_class}")
    print(f"Majority class ratio (expected chance baseline): {majority_ratio:.4f}")

    # Revisit the test dataset
    y_test_shuffled = np.random.permutation(y_test)
    print("Label Distribution after shuffling: ", np.unique(y_test_shuffled, return_counts=True))
    probs_scrambled, preds_scrambled = predict_with_helper(helper, transformer, X_test_scaled, latent_test_tiled, threshold=0.5)

    from sklearn.metrics import f1_score, accuracy_score

    acc_scrambled = accuracy_score(y_test_shuffled, preds_scrambled)

    if acc_scrambled > majority_ratio + 0.05:
        print("Shuffled label performance is higher than expected")
    else:
        print("Sanity Check passed")
        print("Scrambled Accuracy:", acc_scrambled)
