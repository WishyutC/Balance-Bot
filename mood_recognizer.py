import os
from collections import deque
from typing import Any, AnyStr, Callable, NamedTuple
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import warnings

import librosa
import librosa.effects
import librosa.feature
import numpy as np
from numpy.typing import NDArray

from emotion import Emotion

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

os.environ["KERAS_BACKEND"] = "jax"
# noinspection PyPackageRequirements
import keras  # type: ignore
import jax


class AudioFileItem(NamedTuple):
    path: str | os.PathLike[Any]
    label: np.int32


class Model:
    _pending: deque[AudioFileItem] = deque()
    _features: list[NDArray[np.float32]] = []
    _labels: list[np.int32] = []
    _sample_rate: float
    model = None

    def __init__(self, sample_rate: float = 16000):
        self._sample_rate = sample_rate

    def load_dataset(self, root_dir: os.PathLike[AnyStr] | str,
                     code_emotion_map: dict[Any, Emotion],
                     code_derive_function: Callable[[AnyStr], Any]):
        def __scan(dir_path):
            for e in os.scandir(dir_path):
                if e.is_dir():
                    __scan(e.path)
                elif e.is_file():
                    if not e.name.lower().endswith(".wav"):
                        continue
                    try:
                        code = code_derive_function(os.path.basename(e.path))
                        emotion = code_emotion_map[code]
                    except IndexError:
                        raise ValueError(f"Malformed filename: {e.path}")
                    except KeyError:
                        raise ValueError(f"Emotion type not recognized: {e.path}")
                    self._pending.append(AudioFileItem(
                        path=os.path.abspath(e.path),
                        label=np.int32(emotion.value)))

        __scan(root_dir)

    @staticmethod
    def extract_2d_features(y: np.ndarray, sr: float) -> NDArray[np.float32]:
        """Extract 2D mel-spectrogram features for mood recognition"""
        # Ensure consistent audio length (3 seconds)
        target_length = int(sr * 3)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)))

        # Extract mel-spectrogram (2D features)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,  # 128 mel bands
            hop_length=512,
            n_fft=2048
        )

        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize (fix division by zero warning)
        spec_mean = np.mean(log_mel_spec)
        spec_std = np.std(log_mel_spec)
        if spec_std > 1e-8:  # Avoid division by zero
            log_mel_spec = (log_mel_spec - spec_mean) / spec_std
        else:
            log_mel_spec = log_mel_spec - spec_mean  # Just center if std is too small

        return log_mel_spec.astype(np.float32)

    @staticmethod
    def _generate_augmented_audios(y: np.ndarray, sr: float) -> list[NDArray[np.float32]]:
        augmented = []
        # Pitch shifting
        for n_steps in [-1, 1]:
            y_ps = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
            augmented.append(y_ps)
        # Time stretching
        for rate in [0.9, 1.1]:
            y_ts = librosa.effects.time_stretch(y=y, rate=rate)
            # Ensure consistent length
            target_length = len(y)
            if len(y_ts) > target_length:
                y_ts = y_ts[:target_length]
            else:
                y_ts = np.pad(y_ts, (0, target_length - len(y_ts)))
            augmented.append(y_ts)
        # Add noise
        augmented.append(y + np.random.normal(0, 0.005, len(y)))
        return augmented

    def generate_features(self) -> None:
        total = len(self._pending)
        processed = 0
        while self._pending:
            path, label = self._pending.popleft()
            processed += 1
            print(f"Processing {processed}/{total}: {os.path.basename(path)}", end="\r")

            y, sr = librosa.load(path, sr=self._sample_rate)

            # Original audio
            features_2d = self.extract_2d_features(y, sr)
            self._features.append(features_2d)
            self._labels.append(label)

            # Augmented audio
            for augmented_y in self._generate_augmented_audios(y, sr):
                aug_features_2d = self.extract_2d_features(augmented_y, sr)
                self._features.append(aug_features_2d)
                self._labels.append(label)

        print(f"\nCompleted processing {total} audio files with augmentation")

    @staticmethod
    def _process_audio_file_with_augmentation(args):
        """Static method to process a single audio file with augmentation in parallel"""
        path, label, sample_rate = args

        # Load audio
        y, sr = librosa.load(path, sr=sample_rate)

        # Process original and augmented audio
        features_list = []
        labels_list = []

        augmented_audios = [y]
        augmented_audios.extend(Model._generate_augmented_audios(y, sr))
        for aug_audio in augmented_audios:
            features = Model.extract_2d_features(aug_audio, sr)
            features_list.append(features)
            labels_list.append(label)

        return features_list, labels_list

    def generate_features_parallel(self, n_worker=None) -> None:
        if n_worker is None:
            n_worker = min(4, os.cpu_count() or 4)  # Limit workers and avoid None issues

        # Convert deque to list for parallel processing
        pending_items = list(self._pending)
        total = len(pending_items)

        if total == 0:
            print("No audio files to process")
            return

        # Prepare arguments for parallel processing
        args_list = [(item.path, item.label, self._sample_rate) for item in pending_items]

        print(f"Processing {total} audio files with {n_worker} workers...")

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=n_worker) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(self._process_audio_file_with_augmentation, args): i
                for i, args in enumerate(args_list)
            }

            # Collect results as they complete
            processed = 0
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    features_list, labels_list = future.result()
                    # Add all features and labels from this file (including augmented versions)
                    for features, label in zip(features_list, labels_list):
                        self._features.append(features)
                        self._labels.append(label)

                    processed += 1
                    print(f"Processed {processed}/{total} files", end="\r")

                except Exception as exc:
                    index = future_to_index[future]
                    print(f'\nError processing file {args_list[index][0]}: {exc}')

        # Clear the pending queue since we've processed everything
        self._pending.clear()

        print(f"\nCompleted processing {total} audio files with augmentation in parallel")
        print(f"Total features generated: {len(self._features)} (including augmented versions)")

    def save_features_to_path(self, path: str) -> None:
        """Save current features and labels to the given directory path as features.npy and labels.npy"""
        os.makedirs(path, exist_ok=True)
        features_path = os.path.join(path, "features.npy")
        labels_path = os.path.join(path, "labels.npy")
        self.save_features_to_file(features_path, labels_path)
        print(f"Saved features: {len(self._features)} samples to {path}")

    def load_features_from_path(self, path: str) -> bool:
        """Load features and labels from features.npy and labels.npy in the given directory path"""
        features_path = os.path.join(path, "features.npy")
        labels_path = os.path.join(path, "labels.npy")
        if os.path.exists(features_path) and os.path.exists(labels_path):
            self.load_features_from_file(features_path, labels_path)
            print(f"Loaded features: {len(self._features)} total samples from {path}")
            return True
        else:
            print(f"Features not found in {path}")
            return False

    @staticmethod
    def features_exist_in_path(path: str) -> bool:
        """Check if features.npy and labels.npy exist in the given directory path"""
        features_path = os.path.join(path, "features.npy")
        labels_path = os.path.join(path, "labels.npy")
        return os.path.exists(features_path) and os.path.exists(labels_path)

    def process_and_save_dataset_to_path(self, dataset_path: str, emotion_map: dict,
                                         code_function: Callable, path: str, n_worker=0) -> None:
        """Process a dataset and save its features to the given directory path as features.npy and labels.npy"""
        if self.features_exist_in_path(path):
            print(f"Features already exist in {path}, skipping processing...")
            return
        self.clear_features()
        if os.path.exists(dataset_path):
            print(f"Processing dataset: {dataset_path}")
            self.load_dataset(dataset_path, emotion_map, code_function)
            if len(self._pending) > 0:
                if n_worker > 1:
                    self.generate_features_parallel(n_worker=n_worker)
                else:
                    self.generate_features()
                self.save_features_to_path(path)
            else:
                print(f"No audio files found in {dataset_path}")
        else:
            print(f"Dataset path not found: {dataset_path}")

    def load_multiple_features_from_paths(self, paths: list[str]) -> None:
        """Load features from multiple dataset paths (features.npy and labels.npy in each)"""
        self.clear_features()
        for path in paths:
            if self.features_exist_in_path(path):
                self.load_features_from_path(path)
            else:
                print(f"Warning: Features not found in {path}")

    def save_features_to_file(self, features_filename: str | os.PathLike[str],
                              labels_filename: str | os.PathLike[str]) -> None:
        np.save(features_filename, np.array(self._features))
        np.save(labels_filename, np.array(self._labels))

    def load_features_from_file(self, features_filename: str | os.PathLike[str],
                                labels_filename: str | os.PathLike[str]) -> None:
        self._features.extend(np.load(features_filename))
        self._labels.extend(np.load(labels_filename))

    def clear_features(self) -> None:
        self._features.clear()
        self._labels.clear()

    @staticmethod
    def get_available_devices() -> None:
        print("Default backend:", jax.default_backend())
        print("Available devices:", jax.devices())
        """
        physical_devices = tf.config.list_physical_devices()
        logical_devices = tf.config.list_logical_devices()
        print(f"Physical Devices: {len(physical_devices)}, Logical Devices: {len(logical_devices)}")
        for physical_device in physical_devices:
            print(f"P - {physical_device.name}")
        for logical_device in logical_devices:
            print(f"L - {logical_device.name}")
        """

    def init_model(self) -> None:
        keras.backend.clear_session()

        # 2D CNN model for mel-spectrogram features
        # Input shape: (128, time_steps, 1) where 128 is n_mels
        input_shape = self._features[0].shape + (1,) if self._features else (128, 94, 1)

        self.model = keras.models.Sequential([
            keras.layers.Input(shape=input_shape),

            # 2D Convolutional layers
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            # Global pooling and dense layers
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),

            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(len(Emotion), activation='softmax')
        ])

        # noinspection SpellCheckingInspection
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def summary(self):
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")
        self.model.summary()
        print(
            f"Feature shape: {self._features[0].shape if self._features else 'No features loaded'}\n"
            f"Total feature samples: {len(self._features)}\n"
            f"Total labels: {len(self._labels)}\n"
            f"Sample rate: {self._sample_rate}\n"
            f"Using 2D mel-spectrogram features\n"
            f"Approx. model memory: {self.model.count_params() * 4 / 1024 ** 2:.2f} MB"
        )

    def fit(self, evaluate=False, show_plot=False, verbose=2) -> Any:
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")

        # Convert to numpy arrays and add channel dimension for CNN
        x = np.array(self._features, dtype=np.float32)
        x = np.expand_dims(x, axis=-1)  # Add channel dimension
        y = np.array(self._labels, dtype=np.int32)

        fit_history = self.model.fit(
            x,
            y,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_model.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
        )

        if evaluate:
            loss, accuracy = self.evaluate(verbose=verbose)
            print(f"Train loss: {loss}")
            print(f"Train accuracy: {accuracy}")

        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(fit_history.history['loss'], label='loss')
            plt.plot(fit_history.history['val_loss'], label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(fit_history.history['accuracy'], label='accuracy')
            plt.plot(fit_history.history['val_accuracy'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.ylim(0, 1.0)
            plt.show()

        return fit_history

    def evaluate(self, verbose=0) -> tuple[float, float]:
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")
        x = np.array(self._features, dtype=np.float32)
        x = np.expand_dims(x, axis=-1)
        y = np.array(self._labels, dtype=np.float32)
        return self.model.evaluate(x, y, verbose=verbose)

    def predict(self, features: NDArray[np.float32], verbose=0) -> NDArray[np.int32]:
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")

        # Add batch and channel dimensions
        features = np.expand_dims(features, axis=0)  # batch dim
        features = np.expand_dims(features, axis=-1)  # channel dim

        prediction = self.model.predict(features, verbose=verbose)[0]
        if verbose:
            print(prediction)
            for i, prob in enumerate(prediction):
                print(f"{Emotion(i).name}: {prob * 100:.2f}%")
        return prediction

    def predict_from_file(self, f: str | os.PathLike[str], verbose=0) -> NDArray[np.int32]:
        y, sr = librosa.load(f, sr=self._sample_rate)
        features = self.extract_2d_features(y, sr)
        print(f"Predict file: {f}")
        return self.predict(features, verbose=verbose)

    def save_model(self, f: str | os.PathLike[str]) -> None:
        if self.model is None:
            raise ValueError("Model is not initialized. Call init_model() first.")
        self.model.save(f)

    def load_model(self, f: str | bytes | os.PathLike[str]) -> None:
        self.model = keras.saving.load_model(f)
