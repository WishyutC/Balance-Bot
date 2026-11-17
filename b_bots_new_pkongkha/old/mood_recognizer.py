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
from optree import PyTree

from emotion import Emotion

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

os.environ["KERAS_BACKEND"] = "jax"
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
    _n_mfcc: int
    model = None

    def __init__(self, sample_rate: float = 16000, n_mfcc: int = 40):
        self._sample_rate = sample_rate
        self._n_mfcc = n_mfcc  # Keep for backward compatibility

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

    def extract_2d_features(self, y: np.ndarray, sr: float) -> NDArray[np.float32]:
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

        # Helper function to extract features
        def extract_features(audio_data, sr):
            # Ensure consistent audio length (3 seconds)
            target_length = int(sr * 3)
            if len(audio_data) > target_length:
                audio_data = audio_data[:target_length]
            else:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))

            # Extract mel-spectrogram (2D features)
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sr,
                n_mels=128,
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

        # Generate augmented versions
        def generate_augmented_audios(audio_data, sr):
            augmented = [audio_data]  # Include original
            # Pitch shifting
            for n_steps in [-1, 1]:
                y_ps = librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=n_steps)
                augmented.append(y_ps)
            # Time stretching
            for rate in [0.9, 1.1]:
                y_ts = librosa.effects.time_stretch(y=audio_data, rate=rate)
                # Ensure consistent length
                target_length = len(audio_data)
                if len(y_ts) > target_length:
                    y_ts = y_ts[:target_length]
                else:
                    y_ts = np.pad(y_ts, (0, target_length - len(y_ts)))
                augmented.append(y_ts)
            # Add noise
            augmented.append(audio_data + np.random.normal(0, 0.005, len(audio_data)))
            return augmented

        # Process original and augmented audio
        features_list = []
        labels_list = []

        augmented_audios = generate_augmented_audios(y, sr)
        for aug_audio in augmented_audios:
            features = extract_features(aug_audio, sr)
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

    # New methods for separate dataset feature handling
    def save_dataset_features(self, dataset_name: str, base_dir: str = "features") -> None:
        """Save current features with dataset-specific naming"""
        os.makedirs(base_dir, exist_ok=True)
        features_path = os.path.join(base_dir, f"{dataset_name}_features.npy")
        labels_path = os.path.join(base_dir, f"{dataset_name}_labels.npy")
        self.save_features_to_file(features_path, labels_path)
        print(f"Saved {dataset_name} features: {len(self._features)} samples")

    def load_dataset_features(self, dataset_name: str, base_dir: str = "features") -> bool:
        """Load features from a specific dataset"""
        features_path = os.path.join(base_dir, f"{dataset_name}_features.npy")
        labels_path = os.path.join(base_dir, f"{dataset_name}_labels.npy")

        if os.path.exists(features_path) and os.path.exists(labels_path):
            self.load_features_from_file(features_path, labels_path)
            print(f"Loaded {dataset_name} features: {len(self._features)} total samples")
            return True
        else:
            print(f"Features not found for {dataset_name} in {base_dir}")
            return False

    def dataset_features_exist(self, dataset_name: str, base_dir: str = "features") -> bool:
        """Check if features exist for a specific dataset"""
        features_path = os.path.join(base_dir, f"{dataset_name}_features.npy")
        labels_path = os.path.join(base_dir, f"{dataset_name}_labels.npy")
        return os.path.exists(features_path) and os.path.exists(labels_path)

    def process_and_save_dataset(self, dataset_path: str, emotion_map: dict,
                                 code_function: Callable, dataset_name: str,
                                 base_dir: str = "features") -> None:
        """Process a single dataset and save its features separately"""
        if self.dataset_features_exist(dataset_name, base_dir):
            print(f"Features for {dataset_name} already exist, skipping processing...")
            return

        # Clear current features and load this dataset
        self.clear_features()
        if os.path.exists(dataset_path):
            print(f"Processing dataset: {dataset_name} ({dataset_path})")
            self.load_dataset(dataset_path, emotion_map, code_function)

            if len(self._pending) > 0:
                self.generate_features_parallel()
                self.save_dataset_features(dataset_name, base_dir)
            else:
                print(f"No audio files found in {dataset_path}")
        else:
            print(f"Dataset path not found: {dataset_path}")

    def load_multiple_datasets(self, dataset_names: list[str], base_dir: str = "features") -> None:
        """Load features from multiple datasets"""
        self.clear_features()
        for dataset_name in dataset_names:
            if self.dataset_features_exist(dataset_name, base_dir):
                self.load_dataset_features(dataset_name, base_dir)
            else:
                print(f"Warning: Features not found for {dataset_name}")

    # Backward compatibility methods for MFCC interface
    def generate_mfccs_parallel(self, n_worker=None) -> None:
        """Backward compatible method - now generates 2D features"""
        self.generate_features_parallel(n_worker)

    def save_features_to_file(self, features_filename: str | os.PathLike[str],
                              labels_filename: str | os.PathLike[str]) -> None:
        np.save(features_filename, np.array(self._features))
        np.save(labels_filename, np.array(self._labels))

    def save_mfccs_to_file(self, features_filename: str | os.PathLike[str],
                           labels_filename: str | os.PathLike[str]) -> None:
        """Backward compatible method"""
        self.save_features_to_file(features_filename, labels_filename)

    def load_features_from_file(self, features_filename: str | os.PathLike[str],
                                labels_filename: str | os.PathLike[str]) -> None:
        self._features.extend(np.load(features_filename))
        self._labels.extend(np.load(labels_filename))

    def load_mfccs_from_file(self, features_filename: str | os.PathLike[str],
                             labels_filename: str | os.PathLike[str]) -> None:
        """Backward compatible method"""
        self.load_features_from_file(features_filename, labels_filename)

    def clear_features(self) -> None:
        self._features.clear()
        self._labels.clear()

    def clear_mfccs(self) -> None:
        """Backward compatible method"""
        self.clear_features()

    @staticmethod
    def get_available_gpu() -> None:
        print("Default backend:", jax.default_backend())
        print("Available devices:", jax.devices())

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
        X = np.array(self._features, dtype=np.float32)
        X = np.expand_dims(X, axis=-1)  # Add channel dimension
        y = np.array(self._labels, dtype=np.int32)

        fit_history = self.model.fit(
            X,
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
        X = np.array(self._features, dtype=np.float32)
        X = np.expand_dims(X, axis=-1)
        y = np.array(self._labels, dtype=np.float32)
        return self.model.evaluate(X, y, verbose=verbose)

    def predict(self, features: NDArray[np.float32], verbose=0) -> PyTree[Any]:
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

    def predict_from_file(self, f: str | os.PathLike[str], verbose=0) -> PyTree[Any]:
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
