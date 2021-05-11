import spotipy
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data
from sklearn.preprocessing import MinMaxScaler

client_id = "f427c4b4133c421cae669433d503d822"
client_secret = "9ef7ad359ebe492da8bd73054dad6cbc"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

playlist_id='spotify:playlist:4anBPndViYRCuGgbfnKGh7' #insert your playlist id
results = sp.playlist(playlist_id) 

""" saved = sp.current_user_saved_tracks()
print(saved)  """

#LIST GENERATING

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

# check the song feature
features = sp.audio_features(song_meta['id'])
# change dictionary to dataframe
features_df = pd.DataFrame.from_dict(features)

# convert milliseconds to mins
# duration_ms: The duration of the track in milliseconds.
# 1 minute = 60 seconds = 60 × 1000 milliseconds = 60,000 ms
features_df['duration_ms'] = features_df['duration_ms'] / 60000

# combine two dataframe
final_df = song_meta_df.merge(features_df)

pd.set_option('display.max_columns', None)

#RADAR CHART

music_feature=features_df[['danceability','energy','loudness','speechiness',
'acousticness','instrumentalness','liveness','valence','tempo','duration_ms']]

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
plt.savefig('favourite_category.jpg', bbox_inches="tight")

#COUNT PLOT

artist_df = final_df.copy()

descending_order = artist_df['artist'].value_counts().sort_values(ascending=False).index
ax = sb.countplot(y = artist_df['artist'], order=descending_order)

sb.despine(fig=None, ax=None, top=True, right=True, left=False, trim=False)
sb.set(rc={'figure.figsize':(6,7.2)})

ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_title('Songs per Artist Top 10', fontsize=16, fontweight='heavy')
sb.set(font_scale = 1.4)
ax.axes.get_xaxis().set_visible(False)
ax.set_frame_on(False)

y = artist_df['artist'].value_counts()
for i, v in enumerate(y):
    ax.text(v + 0.2, i + .16, str(v), color='black', fontweight='light', fontsize=14)
    
plt.savefig('top10_songs_per_artist.jpg', bbox_inches="tight")


#Box plot

popularity = artist_df['popularity']
artists = artist_df['artist']

plt.figure(figsize=(10,6))

ax = sb.boxplot(x=popularity, y=artists, data=artist_df)
plt.xlim(20,90)
plt.xlabel('Popularity (0-100)')
plt.ylabel('')
plt.title('Song Popularity by Artist', fontweight='bold', fontsize=18)
plt.savefig('top10_artist_popularity.jpg', bbox_inches="tight")



#valami2