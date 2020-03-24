#Automatic data scraping script
#We are doing this in a standard script so that it can be run in the command line, and will take
#less overhead than a jupyter notebook. For the explaination on how this script works and the process
#by which we developed it, refer to the Project Data Aquisition jupyter notebook

#Importing packages
import random
import requests
import json
import base64
import msvcrt
import pandas as pd
import time
import re
from rauth import OAuth2Service
from bs4 import BeautifulSoup

#Setting global vars
GENIUS_MAX_ID = 4227095
GENIUS_MIN_ID = 1

INDEX_FILE = r'data/index.csv'
DATA_FILE = r'data/data.csv'

RATE_LIMIT = .25

SAVE_FREQUENCY = 100

def main():
    #First, check for an indexing file
    try:
        indexing_frame = pd.read_csv(INDEX_FILE)
        id_list = indexing_frame['id_list']
        current_id_index = indexing_frame['current id'][0]
    except IOError:
        #Create an ordered list of all possible Genius IDs
        id_list = [i for i in range(GENIUS_MIN_ID,GENIUS_MAX_ID + 1)]
        #Shuffle the list and set our starting index to 0
        random.shuffle(id_list)
        current_id_index = -1
    
    genius = OAuth2Service(
        client_id = 'f6xD9D1KtMiCZij5I-71axaKAN7G6NsuAtPcXzVLXenKdAyxZvmy9pMFBAvnP3j6',
        client_secret = 'l8ducnbdIpED0rQNKpADr1M5x4_Q4qTuPvXQ7tC5X1p9D-vkiX2uqdQ-oGs9J48jr_sbueHofnB4thJigUrsZA',
        name = 'genius',
        authorize_url = 'https://api.genius.com/oauth/authorize',
        access_token_url = 'https://api.genius.com/oauth/token',
        base_url = 'https://api.genius.com/'
    )
    
    genius_session = genius.get_session('qIcPSKG-IpOYMR0j-Y2NcuHVuGsbGHO3osa4b7BEJuFaBbZgDn26EKjl_whhxSjO')
    
    spotify = OAuth2Service(
        client_id = 'f8ed44dea3354f30b7e49042cfbe1dd1',
        client_secret = 'ecfe526f97074cbf99680e6a09ee58a8',
        name = 'spotify',
        authorize_url = 'https://accounts.spotify.com/authorize',
        access_token_url = 'https://accounts.spotify.com/api/token',
        base_url = 'https://api.spotify.com'
    )
    
    spotify_session = spotify.get_session(get_spotify_token(spotify))
    
    running = True
    start_time = time.time()
    temp_data = []
    
    songs_found = 0
    
    #GET SONGS
    while running:
        print(str(current_id_index) + ' searched and ' + str(songs_found) + ' found'+ ' - Press any key to exit', end='\r')
        if msvcrt.kbhit():
            running = False  
        
        time_took = time.time() - start_time
        if time_took < RATE_LIMIT:
            time.sleep(RATE_LIMIT - time_took)
        start_time = time.time()
        
        current_id_index += 1
        genius_id = id_list[current_id_index]
        
        try:
            genius_call = genius_session.get('songs/' + str(genius_id)).json()
            url = 'https://genius.com' + genius_call['response']['song']['path'] 
            title = remove_post_hyphen(remove_parenthesis(genius_call['response']['song']['title'].strip()))
            artist = genius_call['response']['song']['primary_artist']['name'].strip()
        except KeyError:
            #We accessed an incorrect id
            continue

        spotify_id = None
        #Search spotify by name and name + artist
        for search in [title, title + ' ' + artist]:
            query_params = {'q' : search,
                        'type' : 'track'}
            response = json.loads(spotify_session.get('v1/search', params=query_params).content)
            try:
                spotify_results = response['tracks']['items']
            except KeyError:
                try:
                    print(response['error'])
                    raise RuntimeError
                except KeyError:
                    continue
            if spotify_results == []:
                continue
            for result in spotify_results:
                if verify_song_identity(result, title, artist):
                    spotify_id = result['uri'].split(':')[2]
                    break
                    
        if spotify_id == None:
            #We didn't manage to get a spotify link from the query or genius
            continue
        
        spotify_call = spotify_session.get('/v1/tracks/' + spotify_id).json()
        try:
            hotness = spotify_call['popularity']
            album_id = spotify_call['album']['uri'].split(':')[2]
            title = spotify_call['name']
        except KeyError:
            print('Spotify call failed for the following track')
            print('ID : ' + str(spotify_id) + ' , Name : ' + str(title) + ' , Artist : ' + str(artist))
            raise KeyError
               
        song_page = requests.get(url)
        html = BeautifulSoup(song_page.text,'html.parser')
        lyrics = html.find('div', class_='lyrics').get_text()
        genres = json.loads(html.find('meta', itemprop='page_data')['content'])['dmp_data_layer']['page']['genres']
        
        popularity_page = requests.get('https://t4ils.dev:4433/api/beta/albumPlayCount?albumid=' + album_id)
        playcount = get_playcount(popularity_page, spotify_id)
        
        song = {
            'title' : title,
            'artist' : artist,
            'lyrics' : lyrics,
            'listens' : playcount,
            'hotness' : hotness,
            'genres' : genres,
            'genius ID' : genius_id,
            'spotify ID' : spotify_id
        }
    
        #Reset for the next run
        genius_call = None
        spotify_call = None
        popularity_page = None
        song_page = None
        title = None
        artist = None
        lyrics = None
        playcount = None
        hotness = None
        genres = None
        genius_id = None
        spotify_id = None
        
        temp_data.append(song)
        songs_found += 1
        
        if(songs_found > SAVE_FREQUENCY):
            songs_found = 0
            save(temp_data, current_id_index, id_list)
            temp_data = []
    
    print(str(current_id_index) + ' searched and ' + str(songs_found) + ' found'+ ' - Press any key to exit')
    print('Saving and Closing')
    save(temp_data, current_id_index, id_list)

def get_spotify_token(spotify):
    authorization_id = 'Basic ' + str(base64.b64encode(bytes(spotify.client_id + ':' + spotify.client_secret,'utf-8')),'utf-8')
    header = {'Authorization' : authorization_id}
    body = {'grant_type' : 'client_credentials'}
    return requests.post('https://accounts.spotify.com/api/token', data=body, headers=header).json()['access_token']

def get_playcount(page, song_id):
    for song in json.loads(page.content)['data']:
        if song['uri'] == 'spotify:track:' + song_id:
            return song['playcount']
        
def get_spotify_id_from_genius(json):
    for media in json['response']['song']['media']:
        if media['provider'].lower() == 'spotify':
            return media['native_uri'].split(':')[2]
    return None

def remove_post_hyphen(string):
    return string.split('-')[0].strip()

def remove_parenthesis(string):
    return string.split('(')[0].strip()

def alphise(string):
    return re.sub(r'\W+','',re.sub(r'\d+','',string))

def remove_the(string):
    return string.replace('The ','').strip()

def clean_string(string):
    return alphise(remove_post_hyphen(remove_parenthesis(remove_the(string))))

def verify_song_identity(song, name, artist):
    return (clean_string(song['name'].strip().lower()) == clean_string(name.strip().lower()) and clean_string(song['artists'][0]['name'].strip().lower()) == clean_string(artist.strip().lower()))

def save(temp_data, current_index, id_list):
    
    #Check for a data file to concat to
    try:
        data = pd.read_csv(DATA_FILE)
    except IOError:
        data = pd.DataFrame()
    
    data = pd.concat([data, pd.DataFrame(temp_data)])
    data.to_csv(DATA_FILE, index=False)
    
    indexing_frame = pd.DataFrame({'current id' : current_index, 'id_list' : id_list})
    indexing_frame.to_csv(INDEX_FILE, index=False)
    
if __name__ == '__main__':
    main()