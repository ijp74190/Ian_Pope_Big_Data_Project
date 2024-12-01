import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def check_blocked(row):
    # Filter rows for the same gameId and playId
    same_play = df[(df['gameId'] == row['gameId']) & (df['playId'] == row['playId'])]

    # Check if nflId is in blockedPlayerNFLId1 for rows where position is in ['C', 'G', 'T']
    blockers = same_play[(same_play['position'].isin(['C', 'G', 'T']))]

    # Check if any blockers match the current nflId
    return row['nflId'] in blockers['blockedPlayerNFLId1'].dropna().values


df = pd.read_csv('./BigDataBowl/player_play.csv')

df = df.filter(['gameId','playId','nflId', 'hadDropback',
                'teamAbbr', 'pressureAllowedAsBlocker',
                'blockedPlayerNFLId1'], axis=1)

player_df = pd.read_csv('./BigDataBowl/players.csv')
df = df.merge(player_df[['nflId', 'position']], on='nflId', how='left')


df['blockedPlayerNFLId1'] = df['blockedPlayerNFLId1'].apply(pd.to_numeric, errors='coerce').astype('Int64')
df['wasBlocked'] = df.apply(check_blocked, axis=1)
df['wasBlocked'] = df['wasBlocked'].apply(lambda x: 1 if x else 0)

blocker_counts = df.dropna(subset=['blockedPlayerNFLId1'])
blocker_counts = blocker_counts.groupby(['gameId', 'playId', 'blockedPlayerNFLId1']).size().reset_index(name='blocker_count')

df = df.merge(blocker_counts[['gameId', 'playId', 'blockedPlayerNFLId1', 'blocker_count']],
              on=['gameId', 'playId', 'blockedPlayerNFLId1'], how='left')

df['blocker_count'] = df['blocker_count'] / 4

df['isCenter'] = (df['position'] == 'C').apply(lambda x: 1 if x else 0)
df['isGuard'] = (df['position'] == 'G').apply(lambda x: 1 if x else 0)
df['isTackle'] = (df['position'] == 'T').apply(lambda x: 1 if x else 0)

print(df.head(22))

df.to_csv('./BigDataBowl/player_play_filtered.csv')