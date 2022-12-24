import pandas as pd
import numpy as np

def get_ticker_in_folder(path:str='/Users/andrea/Downloads/Data_CME/', ticker:str='ES', cols:list=None):

    files   = os.listdir( path )
    stacked = []
    for file in tqdm.tqdm(files):
        if file.startswith(ticker):
            print(f'Reading file named {file} ...')
            if cols is None:
                stacked.append( pd.read_csv( os.path.join(path, file), sep=';') )
            else:
                stacked.append(pd.read_csv(os.path.join(path, file), sep=';', usecols=cols))

    return pd.concat( stacked )

def half_hour(x):
    if x >= 30:
        return "30"
    else:
        return "00"

data_path = '/Users/andrea/Downloads/Data_CME/'
data      = 'ESZ22-CME_20221104_155959.txt'

all_data     = get_ticker_in_folder( data_path, 'ES', cols=['Date', 'Time', 'Volume'] )
all_data     = all_data.assign(Hour     = all_data.Time.str[:2].astype(str))
all_data     = all_data.assign(Minute   = all_data.Time.str[3:5].astype(int))
all_data     = all_data.assign(HalfHour = all_data.Hour.str.zfill(2) + all_data.Minute.apply(half_hour))
all_data_gb  = all_data.groupby(['Date', 'HalfHour']).agg({'Volume':'sum'}).reset_index()
all_data_gb_ = all_data.groupby(['HalfHour']).agg({'Volume':'sum'}).reset_index()

es = pd.read_csv( os.path.join(data_path, data), sep=';' )
es = es[es.Price != 0]
es = es.assign(Index    = np.arange(0, es.shape[0], 1))
es = es.assign(Hour     = es.Time.str[:2].astype(str))
es = es.assign(Minute   = es.Time.str[3:5].astype(int))
es = es.assign(HalfHour = es.Hour.str.zfill(2) + es.Minute.apply(half_hour))

max_volume = es.groupby(['HalfHour']).agg({'Volume':'sum'}).reset_index()
plt.bar(max_volume.HalfHour, max_volume.Volume)
plt.xlabel('HalfHour')
plt.ylabel('Volume')
plt.xticks(rotation=90)
plt.tight_layout()

red   = es[(es.Volume >= 10) & (es.TradeType == 1 ) & (es.TotalAskDepth < es.TotalBidDepth ) & (es.BidSize >= 300)] # & (es.Hour < 16)]
green = es[(es.Volume >= 10) & (es.TradeType == 2 ) & (es.TotalAskDepth > es.TotalBidDepth ) & (es.AskSize >= 300)] # & (es.Hour < 16)]

plt.plot(es.Index, es.Price, zorder=0, lw=0.7)
plt.scatter(red.Index, red.Price, color='red', zorder=1, edgecolors='black')
plt.scatter(green.Index, green.Price, color='green', zorder=1, edgecolors='black')
plt.show()