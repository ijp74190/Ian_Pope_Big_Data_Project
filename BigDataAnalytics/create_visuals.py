import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression

players = pd.read_csv('./BigDataBowl/players.csv')
players = players[['nflId', 'position', 'displayName']]
print(players.head())

frames = pd.read_csv('frame_by_frame11.csv')
print(frames.head())

plays = pd.read_csv('person_by_play11.csv')
print(plays.head())

aggregate_counts = pd.read_csv('nflid_group_counts1.csv')
aggregate_counts = pd.merge(aggregate_counts, players, on='nflId', how='inner')
print(aggregate_counts.head())

#Display predicted vs actual pressure rate
#aggregate_counts = aggregate_counts[aggregate_counts['position'] == 'C']
aggregate_counts['predictedPercentage'] = (aggregate_counts['predictedPressure'] / aggregate_counts['totalSnaps'])
aggregate_counts['actualPercentage'] = (aggregate_counts['actualPressure'] / aggregate_counts['totalSnaps'])
aggregate_counts = aggregate_counts[aggregate_counts['totalSnaps'] >= 30]
X = aggregate_counts['predictedPercentage'].values.reshape(-1, 1)
y = aggregate_counts['actualPercentage'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
top_5 = aggregate_counts.nlargest(3, 'predictedPercentage')
bottom_5 = aggregate_counts.nsmallest(3, 'predictedPercentage')
plt.figure(figsize=(10, 6))
plt.scatter(aggregate_counts['predictedPercentage'], aggregate_counts['actualPercentage'], color='blue', alpha=0.5)
plt.plot(aggregate_counts['predictedPercentage'], y_pred, color='red', label='Line of Best Fit', linestyle='-', linewidth=2)
for index, row in top_5.iterrows():
    plt.text(row['predictedPercentage'], row['actualPercentage'], str(row['displayName']),
             fontsize=11, color='black', ha='right', va='bottom')
for index, row in bottom_5.iterrows():
    plt.text(row['predictedPercentage'], row['actualPercentage'], str(row['displayName']),
             fontsize=11, color='black', ha='right', va='bottom')
plt.xlabel('Predicted Pressure Rate')
plt.ylabel('Actual Pressure Rate')
plt.title('Predicted vs Actual Pressure Rate (30 Snap Minimum)')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()


#Best and worst difference
aggregate_counts['difference'] = aggregate_counts['predictedPercentage'] - aggregate_counts['actualPercentage']
top_10 = aggregate_counts.nlargest(5, 'difference')
bottom_10 = aggregate_counts.nsmallest(5, 'difference').iloc[::-1]
top_10_diff = pd.concat([top_10, bottom_10])
table_data = top_10_diff[['displayName', 'predictedPercentage', 'actualPercentage', 'difference']].round(3)
column_names = ['Name', 'Predicted', 'Actual', 'Difference']
plt.figure(figsize=(10,6))
plt.title("Top and Bottom 5 Pressure Rate Difference (Min 50 Snaps)", fontsize=16, fontweight='bold', pad=25)
table = plt.table(cellText=table_data.values,
          colLabels=column_names,
          cellLoc='center', loc='center', colColours=['#f0f0f0']*len(table_data.columns), bbox=[0.0, 0.0, 1.0, 1.0])
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 1.5)
plt.axis('off')
plt.show()


#Best and worst predicted
top_10 = aggregate_counts.nlargest(5, 'predictedPercentage').iloc[::-1]
bottom_10 = aggregate_counts.nsmallest(5, 'predictedPercentage')
top_10_diff = pd.concat([bottom_10, top_10])
table_data = top_10_diff[['displayName', 'predictedPercentage', 'actualPercentage', 'difference']].round(3)
column_names = ['Name', 'Predicted', 'Actual', 'Difference']
plt.figure(figsize=(10,6))
plt.title("Top and Bottom 5 Predicted Pressure Rate (Min 50 Snaps)", fontsize=16, fontweight='bold', pad=25)
table = plt.table(cellText=table_data.values,
          colLabels=column_names,
          cellLoc='center', loc='center', colColours=['#f0f0f0']*len(table_data.columns), bbox=[0.0, 0.0, 1.0, 1.0])
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 1.5)
plt.axis('off')
plt.show()


#Shows single play
game_id = 2022090800
play_id = 122
nfl_id = 42392
filtered_frames = frames[(frames['gameId'] == game_id) & (frames['playId'] == play_id) & (frames['nflId'] == nfl_id)]
# Sort by
filtered_frames = filtered_frames.sort_values(by='frameId')
plt.figure(figsize=(10, 6))
plt.plot(filtered_frames['frameId'], filtered_frames['model_pred_nb'], marker='o', color='b', label='Model Prediction')
plt.axhline(y=0.25, color='r', linestyle='--', label='Predicted Pressure')
plt.xlabel('Frame ID')
plt.ylabel('Pressure Prediction')
plt.title(f'Pressure Prediction for gameId {game_id}, playId {play_id}, nflId {nfl_id}, Mitch Morse, C')
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.ylim(0, 1)
plt.show()


#Confusion matrix
# Compute the confusion matrix
cm = confusion_matrix(plays['pressureAllowedAsBlocker'], plays['model_pred'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Pressure', 'Pressure'],
            yticklabels=['No Pressure', 'Pressure'])
plt.title('Prediction Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()




#Need the datasets to show all data - done
#Need to make a person by play chart - done
#Need to make nflid player name chart and position specific
#Need to relate each visual to name
#Get al players visual