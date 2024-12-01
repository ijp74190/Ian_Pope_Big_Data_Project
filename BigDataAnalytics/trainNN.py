import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./BigDataBowl/complete_filtering_nr.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df_groups = df.groupby(['gameId', 'playId', 'nflId']).ngroup()

train_groups, test_groups = train_test_split(df_groups.unique(), test_size=0.2, random_state=42)

df['is_train'] = df_groups.isin(train_groups)

train_df = df[df['is_train']]
test_df = df[~df['is_train']]

train_df = train_df.drop(columns=['is_train', 'gameId', 'frameId', 'playId', 'nflId'])
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.drop(columns=['is_train', 'gameId', 'frameId', 'playId', 'nflId'])

X_train = train_df.drop(columns=['pressureAllowedAsBlocker'])
y_train = train_df['pressureAllowedAsBlocker']
X_test = test_df.drop(columns=['pressureAllowedAsBlocker'])
y_test = test_df['pressureAllowedAsBlocker']


model = Sequential()
model.add(Dense(128, input_dim=32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC', 'accuracy', 'precision', 'recall'])
model.summary()

class_weight = {0: 1., 1: 1.}
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
          class_weight=class_weight, verbose=1, callbacks=[reduce_lr, early_stopping])

model.save('pressure_allow15.h5')

test_loss, test_auc, test_acc, test_prc, test_rec = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Precision: {test_prc:.4f}")
print(f"Test Recall: {test_rec:.4f}")



# 1. Plot training & validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 2. Plot training & validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 3. Plot training & validation precision
plt.figure(figsize=(12, 6))
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Training and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()

# 4. Plot training & validation recall
plt.figure(figsize=(12, 6))
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Training and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.show()

