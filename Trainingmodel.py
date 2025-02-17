from typing import Sequence
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Load and preprocess data
Standing_df1 = pd.read_csv("StandingCh.txt")

Sitting_df1 = pd.read_csv("SittingCh.txt")

Praying_df1 = pd.read_csv("PrayingCh.txt")

PickPocketing_df1 = pd.read_csv("Pick PocketingCh.txt")

ChainSnatching_df1 = pd.read_csv("Chain SnatchingCh.txt")

ForwardFall_df1 = pd.read_csv("Forward FallCh.txt")

Fighting_df1 = pd.read_csv("FightingCh.txt")

Walking_df1 = pd.read_csv("WalkingBh.txt")

Salute_df1 = pd.read_csv("SaluteCh.txt")

Clapping_df1 = pd.read_csv("ClappingCh.txt")

WavingHand_df1 = pd.read_csv("Waving HandCh.txt")

X = []
y = []
no_of_timesteps = 20

# Add samples for each class
datasets = [Standing_df1, Sitting_df1, Praying_df1, PickPocketing_df1, ChainSnatching_df1, ForwardFall_df1, 
            Fighting_df1, Walking_df1, Salute_df1, Clapping_df1, WavingHand_df1]

for idx, df in enumerate(datasets):
    data_values = df.iloc[:, 1:].values
    n_samples = len(data_values)
    for i in range(no_of_timesteps, n_samples):
        X.append(data_values[i-no_of_timesteps:i, :])
        y.append(idx)

# Convert lists to numpy arrays
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)  # Should output (samples, time_steps, features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CNN-LSTM Model
model = Sequential()
# CNN layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# LSTM layers
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=len(datasets), activation="softmax"))

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("CNN_LSTM_HAR_Model.h5")
