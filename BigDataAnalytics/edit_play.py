import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./BigDataBowl/plays.csv')
df = df.filter(['gameId','playId','quarter', 'down', 'yardsToGo','possessionTeam',
                'defensiveTeam', 'gameClock', 'preSnapHomeScore', 'preSnapVisitorScore',
                'playClockAtSnap''absoluteYardlineNumber', 'offenseFormation', 'playAction',
                'timeToThrow', 'isDropback'], axis=1)

#Filters the file to only include plays that are dropbacks of any kind
df = df[df['isDropback']]
df['quarter'] = df['quarter'] / 5
df['down'] = df['down'] / 4
df['yardsToGo'] = df['yardsToGo'] / df['yardsToGo'].max()
df['gameClock'] = pd.to_timedelta('00:' + df['gameClock']).apply(lambda x: x.total_seconds())
df['gameClock'] = df['gameClock'] / df['gameClock'].max()

#Perhaps go back and look at these rows just to look for anomalies
#Also these are the one hot encoding candidates
df['offenseFormation'] = df['offenseFormation'].fillna('UNKNOWN')
df['homePointLead'] = df['preSnapHomeScore'] - df['preSnapVisitorScore']
df['playAction'] = df['playAction'].apply(lambda x: 1.0 if x else 0.0)

df = pd.get_dummies(df, columns=['offenseFormation'], dtype=float)

df.drop(['isDropback', 'preSnapHomeScore', 'preSnapVisitorScore'], axis=1, inplace=True)

min_value = df['homePointLead'].min()
max_value = df['homePointLead'].max()
df['homePointLead'] = (df['homePointLead'] - min_value) / (max_value - min_value)

print(df.head(22))
df.to_csv('./BigDataBowl/play_filtered.csv')