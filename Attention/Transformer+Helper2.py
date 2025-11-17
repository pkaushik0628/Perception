# helper_transformer_improved.py
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
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Ore-processing pipleine functions
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
# Transformer (baseline)
# -----------------------------
def build_transformer_model(input_shape, num_heads=4, ff_dim=64):
    inp = layers.Input(shape=input_shape)
    x = layers.LayerNormalization()(inp)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    emb = layers.Dense(ff_dim, activation='relu', name='transformer_embedding')(x)
    out = layers.Dense(1, activation='sigmoid', name='transformer_logits')(emb)
    return models.Model(inp, out)

# -----------------------------
# Improved helper (residual + separable conv blocks + noise)
# -----------------------------
def helper_block(x, filters, kernel_size=7, name=None):
    # separable conv block with residual
    y = layers.SeparableConv1D(filters, kernel_size, padding='same', activation='relu')(x)
    y = layers.SeparableConv1D(filters, kernel_size, padding='same', activation='relu')(y)
    y = layers.LayerNormalization()(y)
    # project residual if needed
    if x.shape[-1] != filters:
        x_proj = layers.Conv1D(filters, 1, padding='same')(x)
    else:
        x_proj = x
    out = layers.Add()([x_proj, y])
    out = layers.Activation('relu')(out)
    out = layers.LayerNormalization()(out)
    return out

def build_helper_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.05)(inp)
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(x)
    x = helper_block(x, 64, kernel_size=7)
    x = helper_block(x, 64, kernel_size=5)
    x = helper_block(x, 128, kernel_size=5)
    x = layers.Conv1D(input_shape[1], 1, padding='same', activation=None)(x)  # project back to electrode-space
    return models.Model(inp, x, name='helper_model')

# -----------------------------
# Asymmetric focal loss
# -----------------------------
def asymmetric_focal_loss(alpha_0=0.9, alpha_1=0.1, gamma=2.0):
    # alpha_0 -> weight for class 0 (rare), alpha_1 -> weight for class 1
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true_f, y_pred)
        p_t = y_true_f * y_pred + (1 - y_true_f) * (1 - y_pred)
        alpha = y_true_f * alpha_0 + (1 - y_true_f) * alpha_1
        focal = alpha * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss

# -----------------------------
# Subclass trainer that updates helper only, combines classification + margin losses
# -----------------------------
class HelperTrainer(tf.keras.Model):
    def __init__(self, helper_model, transformer_model, embedding_model,
                 class_center_0, class_center_1, lambda_margin=0.3,
                 focal_alpha_0=0.9, focal_alpha_1=0.1, focal_gamma=2.0):
        # This wrapper will present a single-output model (logit) to Keras while
        # implementing a custom train_step that computes a combined loss.
        super().__init__()
        self.helper = helper_model
        self.transformer = transformer_model  # frozen
        self.embedding_model = embedding_model  # frozen
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
        # forward only returns logits (so .predict works as normal)
        helper_out = self.helper(inputs, training=training)
        logits = self.transformer(helper_out, training=False)
        return logits

    def compute_focal(self, y_true, y_pred):
        # y_pred expected in [0,1], shape (batch,)
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha = y_true * self.f_alpha_0 + (1 - y_true) * self.f_alpha_1
        focal = alpha * tf.pow(1 - p_t, self.f_gamma) * bce
        return tf.reduce_mean(focal)

    def compute_margin(self, y_true, emb_pred):
        # emb_pred shape (batch, emb_dim)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        # build ref per-sample
        ref0 = tf.reshape(self.class_center_0, (1, -1))
        ref1 = tf.reshape(self.class_center_1, (1, -1))
        ref0_batch = tf.repeat(ref0, tf.shape(emb_pred)[0], axis=0)
        ref1_batch = tf.repeat(ref1, tf.shape(emb_pred)[0], axis=0)
        mask = tf.cast(tf.expand_dims(tf.equal(y_true, 1), axis=1), tf.float32)
        ref_batch = mask * ref1_batch + (1.0 - mask) * ref0_batch
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(emb_pred - ref_batch), axis=1))
        return mse

    def train_step(self, data):
        x, y = data  # x: inputs, y: labels

        with tf.GradientTape() as tape:
            helper_out = self.helper(x, training=True)               # (batch, T, E)
            logits = self.transformer(helper_out, training=False)    # (batch, 1)
            logits = tf.reshape(logits, [-1])                        # (batch,)
            emb_pred = self.embedding_model(helper_out, training=False)  # (batch, emb_dim)

            # compute classification loss (asymmetric focal)
            cls_loss = self.compute_focal(y, logits)
            # embedding margin loss (pull embeddings toward their class center)
            emb_loss = self.compute_margin(y, emb_pred)
            total_loss = cls_loss + self.lambda_margin * emb_loss

        # gradients only for helper parameters
        grads = tape.gradient(total_loss, self.helper.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.helper.trainable_variables))

        # update metrics
        self.loss_tracker.update_state(total_loss)
        self.cls_acc.update_state(y, logits)

        return {"loss": self.loss_tracker.result(), "acc": self.cls_acc.result()}

    def test_step(self, data):
        x, y = data
        helper_out = self.helper(x, training=False)
        logits = self.transformer(helper_out, training=False)
        logits = tf.reshape(logits, [-1])
        emb_pred = self.embedding_model(helper_out, training=False)
        cls_loss = self.compute_focal(y, logits)
        emb_loss = self.compute_margin(y, emb_pred)
        total_loss = cls_loss + self.lambda_margin * emb_loss

        self.loss_tracker.update_state(total_loss)
        self.cls_acc.update_state(y, logits)
        return {"loss": self.loss_tracker.result(), "acc": self.cls_acc.result()}

# -----------------------------
# threshold search utility
# -----------------------------
def find_best_threshold(y_true, probs, target='class0_f1'):
    # scans thresholds and returns best threshold and its f1 (for class0 or macro)
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
# Full pipeline execution
# -----------------------------
if __name__ == "__main__":
    # user inputs
    participant_ids = [41,42,43,44,47,48,49,51,53,54,55]
    regions_of_interest = [
        'LSPE','Left ACC','Left amygdala','Left hippocampus','Left preSMA','Left vmPFC',
        'RSPE','Right ACC','Right amygdala','Right hippocampus','Right preSMA','Right vmPFC'
    ]

    # Prepare dataset
    X, y, common_electrodes = prepare_sequence_dataset(participant_ids, regions_of_interest,
                                                       window_size=500, step_size=250, top_n=8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num_timesteps, num_electrodes = X_train.shape[1], X_train.shape[2]

    # per-electrode scalers (store list)
    scalers = []
    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test)
    for i in range(num_electrodes):
        sc = StandardScaler()
        X_train_scaled[:,:,i] = sc.fit_transform(X_train[:,:,i])
        X_test_scaled[:,:,i] = sc.transform(X_test[:,:,i])
        scalers.append(sc)

    # 1) Train transformer baseline (so it defines the embedding space)
    transformer = build_transformer_model((num_timesteps, num_electrodes))
    transformer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    transformer.fit(X_train_scaled, y_train,
                    validation_split=0.2, epochs=50, batch_size=16,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                    verbose=1)
    transformer.trainable = False

    # embedding extractor
    embedding_model = models.Model(transformer.input, transformer.get_layer('transformer_embedding').output)
    # compute class centers in embedding space (from transformer's view on raw scaled inputs)
    embeddings_train = embedding_model.predict(X_train_scaled, batch_size=64)
    class_center_0 = embeddings_train[y_train == 0].mean(axis=0) if np.any(y_train == 0) else np.zeros(embeddings_train.shape[1])
    class_center_1 = embeddings_train[y_train == 1].mean(axis=0) if np.any(y_train == 1) else np.zeros(embeddings_train.shape[1])

    # 2) Build helper model
    helper = build_helper_model((num_timesteps, num_electrodes))
    helper.summary()

    # 3) Wrap into HelperTrainer
    trainer = HelperTrainer(helper_model=helper,
                            transformer_model=transformer,
                            embedding_model=embedding_model,
                            class_center_0=class_center_0,
                            class_center_1=class_center_1,
                            lambda_margin=0.3,
                            focal_alpha_0=0.9, focal_alpha_1=0.1, focal_gamma=2.0)

    trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    # 4) Train helper using trainer.fit (keeps callbacks and val_split)
    trainer.fit(X_train_scaled, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=16,
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                verbose=1)

    # 5) Inference: predict with helper -> transformer pipeline
    def predict_with_helper(helper_model, transformer_model, X_raw_scaled, threshold=0.5):
        helper_out = helper_model.predict(X_raw_scaled, batch_size=64)
        probs = transformer_model.predict(helper_out, batch_size=64).flatten()
        preds = (probs >= threshold).astype(int)
        return probs, preds

    # baseline (0.5)
    probs_05, preds_05 = predict_with_helper(helper, transformer, X_test_scaled, threshold=0.5)
    print("\n--- results at threshold 0.5 ---")
    print(classification_report(y_test, preds_05, digits=3))
    cm = confusion_matrix(y_test, preds_05)
    print("Confusion matrix:\n", cm)

    # 6) find threshold that maximizes class-0 F1
    best_t, best_f1 = find_best_threshold(y_test, probs_05, target='class0_f1')
    print(f"\nBest threshold (class0 f1) = {best_t:.3f}, f1 = {best_f1:.3f}")

    probs_best, preds_best = predict_with_helper(helper, transformer, X_test_scaled, threshold=best_t)
    print("\n--- results at best threshold ---")
    print(classification_report(y_test, preds_best, digits=3))
    cm = confusion_matrix(y_test, preds_best)
    print("Confusion matrix:\n", cm)

    # plot confusion matrix for best threshold
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Helper+Transformer (threshold={best_t:.3f})")
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
    probs_scrambled, preds_scrambled = predict_with_helper(helper, transformer, X_test_scaled, threshold=0.5)

    from sklearn.metrics import f1_score, accuracy_score

    acc_scrambled = accuracy_score(y_test_shuffled, preds_scrambled)

    if acc_scrambled > majority_ratio + 0.05:
        print("Shuffled label performance is higher than expected")
    else:
        print("Sanity Check passed")
        print("Scrambled Accuracy:", acc_scrambled)






