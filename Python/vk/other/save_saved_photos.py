import vk
import os
from urllib.request import urlopen
from time import sleep

session = vk.Session()
api = vk.API(session, v='5.53', lang='ru', timeout=10)

def get_photos_urls(user_id):
    photos_json = api.photos.get(owner_id=user_id, album_id='saved')
    photos_amount = photos_json['count']
    photos_list = photos_json['items']

    result = []
    not_saved = []
    for photo in photos_list:
        if 'photo_604' in photo:
            result.append(photo['photo_604'])
        else:
            try:
                not_saved.append(photo['photo_130'])
            except:
                not_saved.append('ERROR: photo is too small.')

    if len(result) != photos_amount:
        print('Sorry, %i photos are not saved.')
        print('Here are some of them:')
        for photo_url in not_saved:
            print(photo_url)

    return result


def save_photos(photos_urls_list, foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    for i, url in enumerate(photos_urls_list):
        try:
            print('Downloading %s' % url)
            filename = os.path.join(foldername, str(i)+'.jpg')
            print(filename)
            open(filename, 'wb').write(urlopen(url).read())
            sleep(1)
        except:
            continue
    print('Saved!')

save_photos(get_photos_urls(291823738), 'saved_pics')
