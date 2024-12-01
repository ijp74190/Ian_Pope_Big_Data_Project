import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
import numpy as np


def adjust_tracking_data(dataframe):

    df = dataframe

    for game_id in df['gameId'].unique():
        for play_id in df[df['gameId'] == game_id]['playId'].unique():
            play_data = df[(df['gameId'] == game_id) & (df['playId'] == play_id)]

            # Get the play direction
            play_direction = play_data['playDirection'].iloc[0]

            if play_direction == 'left':
                # Flip the x position
                df.loc[(df['gameId'] == game_id) & (df['playId'] == play_id), 'x'] = 120 - df['x']

                # Flip the y position
                df.loc[(df['gameId'] == game_id) & (df['playId'] == play_id), 'y'] = 53.3 - df['y']

                # Flip the direction and orientation
                df.loc[(df['gameId'] == game_id) & (df['playId'] == play_id), 'dir'] = (df['dir'] + 180) % 360
                df.loc[(df['gameId'] == game_id) & (df['playId'] == play_id), 'o'] = (df['o'] + 180) % 360

    return df


columns = ['gameId', 'playId', 'frameId', 'nflId', 'quarter', 'down', 'yardsToGo', 'gameClock', 'playAction',
        'homePointLead', 'offenseFormation_EMPTY', 'offenseFormation_I_FORM', 'offenseFormation_JUMBO',
        'offenseFormation_PISTOL', 'offenseFormation_SHOTGUN', 'offenseFormation_SINGLEBACK',
        'offenseFormation_UNKNOWN', 'offenseFormation_WILDCAT', 'hadDropback', 'blocker_count',
        'isCenter', 'isGuard', 'isTackle', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'passOut',
        'pointDifferential', 'distanceToQb', 'distanceToDef', 'defToQB', 'beforeSnap', 'pressureAllowedAsBlocker']
total_df = pd.DataFrame(columns=columns)


for i in range(1,10):
    path = './BigDataBowl/tracking_week_' + str(i) + '.csv'
    tracking = pd.read_csv(path)

    plays = pd.read_csv('./BigDataBowl/play_filtered.csv')
    player_plays = pd.read_csv('./BigDataBowl/player_play_filtered.csv')

    plays = plays.loc[:, ~plays.columns.str.contains('^Unnamed')]
    player_plays = player_plays.loc[:, ~player_plays.columns.str.contains('^Unnamed')]
    tracking = tracking.loc[:, ~tracking.columns.str.contains('^Unnamed')]

    df = pd.read_csv('./BigDataBowl/games.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df[df['week'] == i]
    df = pd.merge(df, plays, on=['gameId'], how='left')

    df = pd.merge(df, player_plays, on=['gameId', 'playId'], how='left')

    df = pd.merge(df, tracking, on=['gameId', 'playId', 'nflId'], how='left')

    df = df.sort_values(by=['gameId', 'playId', 'frameId'])
    pass_frame_idx = df[df['event'] == 'pass_forward'].groupby(['gameId', 'playId'])['frameId'].transform('min')
    df['pass_frame_idx'] = pass_frame_idx
    df['passOut'] = (df['frameId'] >= df['pass_frame_idx']).astype(int)

    df = df[(df['hadDropback'] == 1) | (df['wasBlocked'] == 1) | (df['isCenter'] == 1) |
            (df['isGuard'] == 1) | (df['isTackle'] == 1)]

    df['pointDifferential'] = df.apply(lambda row: row['homePointLead']
    if row['teamAbbr'] == row['homeTeamAbbr'] else -row['homePointLead'], axis=1)

    # Flip values so offense always going right
    df = adjust_tracking_data(df)

    # Get distance to qb
    qb_df = df[df['hadDropback'] == 1][['gameId', 'playId', 'frameId', 'x', 'y']]
    ol_df = df[(df['isCenter'] == 1) | (df['isGuard'] == 1) | (df['isTackle'] == 1)]
    merged_df = ol_df.merge(qb_df, on=['gameId', 'playId', 'frameId'], suffixes=('', '_qb'))
    merged_df['distanceToQb'] = np.sqrt(
        (merged_df['x'] - merged_df['x_qb']) ** 2 + (merged_df['y'] - merged_df['y_qb']) ** 2)
    df = df.merge(merged_df[['nflId', 'gameId', 'playId', 'frameId', 'distanceToQb', 'x_qb', 'y_qb']],
                  on=['nflId', 'gameId', 'playId', 'frameId'], how='left')

    # Get distance to defender
    blocked_df = df[df['blockedPlayerNFLId1'].notna()]
    merged_df = blocked_df.merge(df,
                                 left_on=['gameId', 'playId', 'frameId', 'blockedPlayerNFLId1'],
                                 right_on=['gameId', 'playId', 'frameId', 'nflId'],
                                 suffixes=('_blocker', '_blocked'))
    merged_df['distanceToDef'] = np.sqrt((merged_df['x_blocker'] - merged_df['x_blocked']) ** 2 +
                                         (merged_df['y_blocker'] - merged_df['y_blocked']) ** 2)
    df = df.merge(merged_df[['nflId_blocker', 'gameId', 'playId', 'frameId', 'distanceToDef', 'x_blocked', 'y_blocked']],
                  left_on=['nflId', 'gameId', 'playId', 'frameId'],
                  right_on=['nflId_blocker', 'gameId', 'playId', 'frameId'],
                  how='left')

    # Drop all non lineman
    df = df[(df['isCenter'] == 1) | (df['isGuard'] == 1) | (df['isTackle'] == 1)]

    df['defToQB'] = df.apply(lambda row: np.sqrt((row['x_blocked'] - row['x_qb']) ** 2 +
                                                 (row['y_blocked'] - row['y_qb']) ** 2), axis=1)

    df['beforeSnap'] = df.apply(lambda row: 1 if row['frameType'] == 'BEFORE_SNAP' else 0, axis=1)

    min_value = df['x'].min()
    max_value = df['x'].max()
    df['x'] = (df['x'] - min_value) / (max_value - min_value)
    min_value = df['y'].min()
    max_value = df['y'].max()
    df['y'] = (df['y'] - min_value) / (max_value - min_value)
    min_value = df['s'].min()
    max_value = df['s'].max()
    df['s'] = (df['s'] - min_value) / (max_value - min_value)
    min_value = df['a'].min()
    max_value = df['a'].max()
    df['a'] = (df['a'] - min_value) / (max_value - min_value)
    min_value = df['dis'].min()
    max_value = df['dis'].max()
    df['dis'] = (df['dis'] - min_value) / (max_value - min_value)
    min_value = df['o'].min()
    max_value = df['o'].max()
    df['o'] = (df['o'] - min_value) / (max_value - min_value)
    min_value = df['dir'].min()
    max_value = df['dir'].max()
    df['dir'] = (df['dir'] - min_value) / (max_value - min_value)
    min_value = df['distanceToQb'].min()
    max_value = df['distanceToQb'].max()
    df['distanceToQb'] = (df['distanceToQb'] - min_value) / (max_value - min_value)
    min_value = df['distanceToDef'].min()
    max_value = df['distanceToDef'].max()
    df['distanceToDef'] = (df['distanceToDef'] - min_value) / (max_value - min_value)
    min_value = df['defToQB'].min()
    max_value = df['defToQB'].max()
    df['defToQB'] = (df['defToQB'] - min_value) / (max_value - min_value)

    df = df[['gameId', 'playId', 'frameId', 'nflId', 'quarter', 'down', 'yardsToGo', 'gameClock', 'playAction',
             'homePointLead', 'offenseFormation_EMPTY', 'offenseFormation_I_FORM', 'offenseFormation_JUMBO',
             'offenseFormation_PISTOL', 'offenseFormation_SHOTGUN', 'offenseFormation_SINGLEBACK',
             'offenseFormation_UNKNOWN', 'offenseFormation_WILDCAT', 'hadDropback', 'blocker_count',
             'isCenter', 'isGuard', 'isTackle', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'passOut',
             'pointDifferential', 'distanceToQb', 'distanceToDef', 'defToQB', 'beforeSnap', 'pressureAllowedAsBlocker']]

    df = df.dropna()

    print("appending", i)
    total_df = pd.concat([total_df, df], ignore_index=True)
    print("done with", i)


print("saving file")
total_df.to_csv('./BigDataBowl/complete_filtering_nr.csv')
print("done")
print(total_df.head())