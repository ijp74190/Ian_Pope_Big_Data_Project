import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

model = load_model('./pressure_allow1.h5')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Loading Data")


df = pd.read_csv('./BigDataBowl/complete_filtering_nr.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Create identifier based on gameId, playId, and nflId
df_groups = df.groupby(['gameId', 'playId', 'nflId']).ngroup()

# Split data into train and test based on the groups
#train_groups, test_groups = train_test_split(df_groups.unique(), test_size=0.2, random_state=42)
#df['is_train'] = df_groups.isin(train_groups)

#train_df = df[df['is_train']]
#test_df = df[~df['is_train']]
test_df = df

# Process for pressureAllowedAsBlocker == 1
filtered_train_df = test_df[test_df['pressureAllowedAsBlocker'] == 1]
unique_combinations_allowed = filtered_train_df[['gameId', 'playId', 'nflId']].drop_duplicates()
unique_count_allowed = unique_combinations_allowed.shape[0]

# Process for pressureAllowedAsBlocker != 1
filtered_not_train_df = test_df[test_df['pressureAllowedAsBlocker'] != 1]
unique_combinations_not_allowed = filtered_not_train_df[['gameId', 'playId', 'nflId']].drop_duplicates()
unique_count_not_allowed = unique_combinations_not_allowed.shape[0]

print(f"Unique count where pressureAllowedAsBlocker == 1: {unique_count_allowed}")
print(f"Unique count where pressureAllowedAsBlocker != 1: {unique_count_not_allowed}")

# Prepare data for testing
X_test = test_df.drop(columns=['pressureAllowedAsBlocker', 'gameId', 'frameId', 'playId', 'nflId'])
#X_test = test_df.drop(columns=['pressureAllowedAsBlocker', 'is_train', 'gameId', 'frameId', 'playId', 'nflId'])
y_test = test_df['pressureAllowedAsBlocker']

print("Data processing complete")

# Make predictions
y_pred = model.predict(X_test)
# 51 = .5 x = 0.675 5 = .55
y_pred_binary = (y_pred > 0.25).astype(int)

cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Pressure', 'Pressure'],
            yticklabels=['No Pressure', 'Pressure'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Add back columns for grouping purposes
test_df['model_pred'] = y_pred_binary
test_df['model_pred_nb'] = y_pred

# Group by ['gameId', 'playId', 'nflId'] and check if any row has a 1 in 'model_pred' or 'pressureAllowedAsBlocker'
grouped_predictions = test_df.groupby(['gameId', 'playId', 'nflId']).agg(
    model_pred=('model_pred', 'max'),
    pressureAllowedAsBlocker=('pressureAllowedAsBlocker', 'max')
).reset_index()
print(grouped_predictions.head())
grouped_predictions.to_csv('person_by_play.csv', index=False)
print("Grouped predictions saved to 'person_by_play.csv'")


# Group the test set by ['gameId', 'playId', 'nflId']
grouped_test = test_df.groupby(['gameId', 'playId', 'nflId'])

game_play_nflid_with_pred_1 = []

total_groups_by_nflid = {}
groups_with_pred_1_by_nflid = {}
pressure_allowed_counts_by_nflid = {}

for (gameId, playId, nflId), group in grouped_test:
    # Total Plays per player
    if nflId not in total_groups_by_nflid:
        total_groups_by_nflid[nflId] = 0
    total_groups_by_nflid[nflId] += 1

    # Actual pressure count aggregated
    # Count how many times pressureAllowedAsBlocker == 1 for the current nflId
    pressure_allowed_count = group['pressureAllowedAsBlocker'].sum()  # Sum gives the count of 1s
    if nflId not in pressure_allowed_counts_by_nflid:
        pressure_allowed_counts_by_nflid[nflId] = 0
    pressure_allowed_counts_by_nflid[nflId] += 1 if pressure_allowed_count != 0 else 0

    # Check if any row in the group has a model_pred of 1
    if (group['model_pred'] == 1).any():
        # All 1 predictions
        game_play_nflid_with_pred_1.append((gameId, playId, nflId))

        # Count of predicted yes
        if nflId not in groups_with_pred_1_by_nflid:
            groups_with_pred_1_by_nflid[nflId] = 0
        groups_with_pred_1_by_nflid[nflId] += 1


game_play_nflid_with_pred_1_normal = [(int(gameId), int(playId), int(nflId)) for gameId, playId, nflId in game_play_nflid_with_pred_1]
print(f"First play in list to predict 1: {game_play_nflid_with_pred_1_normal[0]}")
print(f"Total number of combinations where model predicted 1: {len(game_play_nflid_with_pred_1)}")

data_to_save = []
for nflId in total_groups_by_nflid:
    total_count = total_groups_by_nflid[nflId]
    pred_1_count = groups_with_pred_1_by_nflid.get(nflId, 0)
    pressure_allowed_count = pressure_allowed_counts_by_nflid.get(nflId, 0)
    data_to_save.append([nflId, total_count, pred_1_count, pressure_allowed_count])

df_result = pd.DataFrame(data_to_save, columns=['nflId', 'totalSnaps', 'predictedPressure', 'actualPressure'])
print(df_result.head())

df_result.to_csv('nflid_group_counts.csv', index=False)

print("Results saved to 'nflid_group_counts.csv'")

print('saving frame by frame')
test_df = test_df[['gameId', 'playId', 'frameId', 'nflId', 'model_pred_nb', 'pressureAllowedAsBlocker']]
print(test_df.head())
test_df.to_csv('frame_by_frame.csv', index=False)

print('done')