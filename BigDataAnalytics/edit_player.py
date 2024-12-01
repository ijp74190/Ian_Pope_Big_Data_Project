import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('./BigDataBowl/players.csv')

df = df.filter(['nflId', 'position'], axis=1)

print(df['position'].unique())
print(df.head())
#['QB' 'T' 'TE' 'WR' 'DE' 'NT' 'SS' 'FS' 'G' 'OLB' 'DT'
# 'CB' 'RB' 'C' 'ILB' 'MLB' 'FB' 'DB' 'LB']