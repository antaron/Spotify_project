import spotipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data
from sklearn.preprocessing import MinMaxScaler
#%%Adatok beolvasása
client_id = "f427c4b4133c421cae669433d503d822"
client_secret = "9ef7ad359ebe492da8bd73054dad6cbc"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

playlist_id='spotify:playlist:01uAr22ptrHaSQSrVlgsT5' #insert your playlist id
results = sp.playlist(playlist_id) 

""" saved = sp.current_user_saved_tracks()
print(saved)  """
#%%Lista generálás

# create a list of song ids
ids = []

for item in results['tracks']['items']:
    track = item['track']['id']
    ids.append(track)

song_meta = {'id': [], 'album': [], 'name': [],
             'artist': [], 'explicit': [], 'popularity': []}

for song_id in ids:
    # get song's meta data
    meta = sp.track(song_id)

    # song id
    song_meta['id'].append(song_id)

    # album name
    album = meta['album']['name']
    song_meta['album'] += [album]

    # song name
    song = meta['name']
    song_meta['name'] += [song]

    # artists name
    s = ', '
    artist = s.join([singer_name['name'] for singer_name in meta['artists']])
    song_meta['artist'] += [artist]

    # explicit: lyrics could be considered offensive or unsuitable for children
    explicit = meta['explicit']
    song_meta['explicit'].append(explicit)

    # song popularity
    popularity = meta['popularity']
    song_meta['popularity'].append(popularity)

song_meta_df = pd.DataFrame.from_dict(song_meta)
#%%
# song feature
features = sp.audio_features(song_meta['id'])
# change dictionary to dataframe
features_df = pd.DataFrame.from_dict(features)

# convert milliseconds to mins
# duration_ms: The duration of the track in milliseconds.
# 1 minute = 60 seconds = 60 × 1000 milliseconds = 60,000 ms
features_df['duration_ms'] = features_df['duration_ms'] / 60000

#%% combine two dataframe - final data létrehozása
final_df = song_meta_df.merge(features_df)

pd.set_option('display.max_columns', None)

artist_df = final_df.copy()

#%%Elemzés 
elemzes = final_df.describe()
final_df.info()

#%% Korrelációs mátrix 
plt.figure(figsize=(8,8))
sns.heatmap(final_df.corr(),annot=True)
#%%RADAR CHART

music_feature=features_df[['danceability','energy','loudness','speechiness',
'acousticness','instrumentalness','liveness','valence','tempo','duration_ms']]

feature_vizsgalat=features_df[['danceability','energy','speechiness',
'acousticness','liveness','valence']]
min_max_scaler = MinMaxScaler()
music_feature.loc[:]=min_max_scaler.fit_transform(music_feature.loc[:])

# plot size
fig=plt.figure(figsize=(12,8))

# convert column names into a list
categories=list(music_feature.columns)
# number of categories
N=len(categories)

# create a list with the average of all features
value=list(music_feature.mean())

# repeat first value to close the circle
# the plot is a circle, so we need to "complete the loop"
# and append the start value to the end.
pi = np.pi
value+=value[:1]
# calculate angle for each category
angles=[n/float(N)*2*pi for n in range(N)]
angles+=angles[:1]

# plot
plt.polar(angles, value)
plt.fill(angles,value,alpha=0.3)

# plt.title('Discovery Weekly Songs Audio Features', size=35)

plt.xticks(angles[:-1],categories, size=15)
plt.yticks(color='grey',size=15)
#plt.savefig('favourite_category.jpg', bbox_inches="tight")

plt.title('Milyen feature-ök a legfontosabbak egy számban?', fontweight='bold', fontsize=16)
#%% Összehasonlítandó értékek elkészítése
elemzes2 = feature_vizsgalat.mean()
elemzes3 = elemzes2.sort_values(0,ascending = False).head(2)

#%% 
elemzes3 = elemzes3.to_frame()

#%%
elemzes3 = elemzes3.transpose()
#%%
elemzes3['First_max'] = 0
elemzes3['First_min'] = 0
elemzes3['Second_max'] = 0
elemzes3['Second_min'] = 0

elemzes3['First_max'] = elemzes3.iloc[0, 0] + 0.1
elemzes3['First_min'] = elemzes3.iloc[0, 0] - 0.1
elemzes3['Second_max'] = elemzes3.iloc[0, 1] + 0.1
elemzes3['Second_min'] = elemzes3.iloc[0, 1] - 0.1

first = elemzes3.columns[0]
second = elemzes3.columns[1]



#%%Mitől lesz jó egy szám?
fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
fig.text(0.5, 0.04, 'Érték', ha='center',size=20)
fig.text(0.04, 0.5, 'Gyakoriság', va='center', rotation='vertical',size=20)
axs[0].hist(final_df['danceability'])
axs[0].set_title('Danceability')
axs[1].hist(final_df['energy'])
axs[1].set_title('Energy')
axs[2].hist(final_df['liveness'])
axs[2].set_title('Liveness')
axs[3].hist(final_df['acousticness'])
axs[3].set_title('Acousticness')

fig.suptitle('Mitől lesz jó egy szám?',size=20)

plt.show()

#%% Explicit chart
labels= final_df['explicit'].value_counts()
plt.figure(figsize=(12, 7))
final_df['explicit'].value_counts().plot(kind='pie',
             autopct='%1.0f%%', labeldistance=1.2, colors = ['#2CCBC0', '#4B58A9'] )

plt.title('Mennyi explicit tartalom van a listán?', fontweight='bold', fontsize=16)

#%% Top 20 előadó
artist_new = artist_df['artist']

df3 = artist_new.to_frame()
df4 = pd.concat([df3['artist'].str.split(', ', expand=True)], axis=1)

all_values = []
for column in df4:
    this_column_values = df4[column].tolist()
    all_values += this_column_values

one_column_df = pd.DataFrame(all_values)

new_df = one_column_df.mask(one_column_df.eq('None')).dropna()

new_df.rename(columns={0:'előadó'},inplace=True)

new_df['count'] = new_df['előadó'].map(new_df['előadó'].value_counts())
new = new_df.drop_duplicates()

new2 = new.sort_values('count',ascending = False).head(20)

plt.figure(figsize=(12, 7))
ax= sns.barplot(x='count', y="előadó", data=new2, palette=("Blues_d"))
ax.set_xlabel('Előfordulás a listában')
ax.set_ylabel('Előadó neve')
ax.set_title('Top 20 előadó')



#%% Global 50
playlist_id_global='spotify:playlist:37i9dQZEVXbMDoHDwVN2tF' #insert your playlist id
results_global = sp.playlist(playlist_id_global) 

""" saved = sp.current_user_saved_tracks()
print(saved)  """
#%%Lista generálás

# create a list of song ids
ids = []

for item in results_global['tracks']['items']:
    track = item['track']['id']
    ids.append(track)

song_meta = {'id': [], 'album': [], 'name': [],
             'artist': [], 'explicit': [], 'popularity': []}

for song_id in ids:
    # get song's meta data
    meta = sp.track(song_id)

    # song id
    song_meta['id'].append(song_id)

    # album name
    album = meta['album']['name']
    song_meta['album'] += [album]

    # song name
    song = meta['name']
    song_meta['name'] += [song]

    # artists name
    s = ', '
    artist = s.join([singer_name['name'] for singer_name in meta['artists']])
    song_meta['artist'] += [artist]

    # explicit: lyrics could be considered offensive or unsuitable for children
    explicit = meta['explicit']
    song_meta['explicit'].append(explicit)

    # song popularity
    popularity = meta['popularity']
    song_meta['popularity'].append(popularity)

song_meta_df = pd.DataFrame.from_dict(song_meta)
#%%
# check the song feature
features_global = sp.audio_features(song_meta['id'])
# change dictionary to dataframe
features_df_global = pd.DataFrame.from_dict(features_global)

# convert milliseconds to mins
# duration_ms: The duration of the track in milliseconds.
# 1 minute = 60 seconds = 60 × 1000 milliseconds = 60,000 ms
features_df_global['duration_ms'] = features_df_global['duration_ms'] / 60000

#%% combine two dataframe - final data létrehozása
global_final_df = song_meta_df.merge(features_df_global)

feature_vizsgalat_global=global_final_df[['album','artist','name','danceability','energy','speechiness',
'acousticness','liveness','valence']]

spike_cols = [col for col in feature_vizsgalat_global.columns if first in col]

feature_vizsgalat_global.loc[(feature_vizsgalat_global[first] < elemzes3.iloc[0, 2]) & (feature_vizsgalat_global[first] > elemzes3.iloc[0, 3]) & (feature_vizsgalat_global[second] < elemzes3.iloc[0, 4]) & (feature_vizsgalat_global[second] > elemzes3.iloc[0, 5]) , 'recommended'] = "Teljes egyezés"
feature_vizsgalat_global.loc[((feature_vizsgalat_global[first] < elemzes3.iloc[0, 2]) & (feature_vizsgalat_global[first] > elemzes3.iloc[0, 3])) | ((feature_vizsgalat_global[second] < elemzes3.iloc[0, 4]) & (feature_vizsgalat_global[second] > elemzes3.iloc[0, 5])) & ((feature_vizsgalat_global[first] < elemzes3.iloc[0, 2]) & (feature_vizsgalat_global[first] > elemzes3.iloc[0, 3]) & (feature_vizsgalat_global[second] < elemzes3.iloc[0, 4]) & (feature_vizsgalat_global[second] > elemzes3.iloc[0, 5])) == False , "recommended"] = "Részleges egyezés"
feature_vizsgalat_global.loc[feature_vizsgalat_global["recommended"].isnull(), "recommended"] = "Nincs egyezés"

#%%
feature_vizsgalat_global['track'] = feature_vizsgalat_global['artist'] + ': '+ feature_vizsgalat_global['name']

#%%
pd.set_option("max_colwidth", 100)
ajanlott =feature_vizsgalat_global[feature_vizsgalat_global['recommended']=='Teljes egyezés']['track']
print('Kifejezetten ajánlott: ', ajanlott.to_string(index=False))

kozepes =feature_vizsgalat_global[feature_vizsgalat_global['recommended']=='Részleges egyezés']['track']
print('Közepesen ajánlott: ', kozepes.to_string(index=False))

nemajanlott =feature_vizsgalat_global[feature_vizsgalat_global['recommended']=='Nincs egyezés']['track']
print('Nem ajánlott: ', nemajanlott.to_string(index=False))

#%%
plt.figure(figsize=(12, 7))
bx = sns.countplot(x="recommended", data=feature_vizsgalat_global, palette="Set3")
bx.set_xlabel('Egyezés')
bx.set_ylabel('Track-ek száma')
bx.set_title('Egyezések eloszlása')
