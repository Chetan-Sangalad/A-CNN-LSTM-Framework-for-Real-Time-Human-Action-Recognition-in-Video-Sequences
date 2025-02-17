from typing import Sequence
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

Standing_df1 = pd.read_csv("StandingCh.txt")
# Standing_df2 = pd.read_csv("Standingbho.txt")
# Standing_df3 = pd.read_csv("Standingch.txt")
# Standing_df4 = pd.read_csv("Standingche.txt")

Sitting_df1 = pd.read_csv("SittingCh.txt")
# Sitting_df2 = pd.read_csv("Sittingbho.txt")
# Sitting_df3 = pd.read_csv("Sittingch.txt")
# Sitting_df4 = pd.read_csv("Sittingche.txt")

Praying_df1 = pd.read_csv("PrayingCh.txt")
# Praying_df2 = pd.read_csv("Prayingbho.txt")
# Praying_df3 = pd.read_csv("Prayingch.txt")
# Praying_df4 = pd.read_csv("Prayingche.txt")

PickPocketing_df1 = pd.read_csv("Pick PocketingCh.txt")
# PickPocketing_df2 = pd.read_csv("pickPocketingbho.txt")
# PickPocketing_df3 = pd.read_csv("pickPocketingch.txt")
# PickPocketing_df4 = pd.read_csv("pickPocketingche.txt")

ChainSnatching_df1 = pd.read_csv("Chain SnatchingCh.txt")
# ChainSnatching_df2 = pd.read_csv("ChainSnatchingbho.txt")
# ChainSnatching_df3 = pd.read_csv("ChainSnatchingch.txt")
# ChainSnatching_df4 = pd.read_csv("ChainSnatchingche.txt")

ForwardFall_df1 = pd.read_csv("Forward FallCh.txt")
# ForwardFall_df2 = pd.read_csv("ForwradFallbho.txt")
# ForwardFall_df3 = pd.read_csv("ForwradFallch.txt")
# ForwardFall_df4 = pd.read_csv("ForwradFallche.txt")

Fighting_df1 = pd.read_csv("FightingCh.txt")
# Fighting_df2 = pd.read_csv("Fightingbho.txt")
# Fighting_df3 = pd.read_csv("Fightingch.txt")
# Fighting_df4 = pd.read_csv("Fightingche.txt")

Walking_df1 = pd.read_csv("WalkingBh.txt")
# Walking_df2 = pd.read_csv("Walkingbho.txt")
# Walking_df3 = pd.read_csv("Walkingch.txt")
# Walking_df4 = pd.read_csv("Walkingche.txt")
Salute_df1 = pd.read_csv("SaluteCh.txt")
# Salute_df2 = pd.read_csv("Salutech.txt")
# Salute_df3 = pd.read_csv("Saluteche.txt")

Clapping_df1 = pd.read_csv("ClappingCh.txt")
# Clapping_df2 = pd.read_csv("Clappingch.txt")
# Clapping_df3 = pd.read_csv("Clappingbh.txt")

WavingHand_df1 = pd.read_csv("Waving HandCh.txt")
# WavingHand_df2 = pd.read_csv("Waving Handch.txt")
# WavingHand_df3 = pd.read_csv("Waving Handche.txt")

# Jumping_df1 = pd.read_csv("JumpingCh.txt")
# Jumping_df2 = pd.read_csv("Jumpingch.txt")
# Jumping_df3 = pd.read_csv("Jumpingvh.txt")
# Jumping_df4 = pd.read_csv("Jumpingsh.txt")

X = []
y = []
no_of_timesteps = 20

datasets = Standing_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0)

# datasets = Standing_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(1)

# datasets = Standing_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(2)

# datasets = Standing_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(3)

datasets = Sitting_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(1) 

# datasets = Sitting_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(5) 

# datasets = Sitting_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(6) 

# datasets = Sitting_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(7) 

datasets = Praying_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(2)  

# datasets = Praying_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(9) 

# datasets = Praying_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(10) 

# datasets = Praying_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(11)

datasets = PickPocketing_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(3)  

# datasets = PickPocketing_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(13)  

# datasets = PickPocketing_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(14)  

# datasets = PickPocketing_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(15)  

datasets = ChainSnatching_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(4) 

# datasets = ChainSnatching_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(17)  

# datasets = ChainSnatching_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(18)  

# datasets = ChainSnatching_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(19)

datasets = ForwardFall_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(5)

# datasets = ForwardFall_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(21)    


# datasets = ForwardFall_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(22)    


# datasets = ForwardFall_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(23)    


datasets = Fighting_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(6)  


# datasets = Fighting_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(25)  

# datasets = Fighting_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(26)  

# datasets = Fighting_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(27)  

datasets = Walking_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(7) 

# datasets = Walking_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(29) 

# datasets = Walking_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(30) 

# datasets = Walking_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(31) 

datasets = Salute_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(8)  

# datasets = Salute_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(29) 

# datasets = Salute_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(30)  

datasets = Clapping_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(9)  

# datasets = Clapping_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(32) 

# datasets = Clapping_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(33) 

datasets = WavingHand_df1.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(10) 

# datasets = WavingHand_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(35) 

# datasets = WavingHand_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(36) 

# datasets = Jumping_df1.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(11) 

# datasets = Jumping_df2.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(38) 

# datasets = Jumping_df3.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(39) 

# datasets = Jumping_df4.iloc[:, 1:].values
# n_samples = len(datasets)
# for i in range(no_of_timesteps, n_samples):
#     X.append(datasets[i-no_of_timesteps:i, :])
#     y.append(40) 

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=11, activation="softmax"))  

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

model.save("FinalProject.h5")
