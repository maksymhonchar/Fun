import vk
import requests
import json
from time import sleep

session = vk.Session(access_token='s4cre1')
api = vk.API(session)

# testing vk serv.
print(api.getServerTime())


# Album for testing: 235220146
def post_photo(album_id, photo='test.png'):
    # mid - my id
    # aid - album id
    # upload_url - server link
    serv_addr = api.photos.getUploadServer(album_id=album_id)
    files = {
        'photo': (
            'test.jpg', open(photo, 'rb')
        )
    }
    url = serv_addr['upload_url']
    data = {
        'gid': "0",
        'act': 'do_add',
        'mid': serv_addr['mid'],
        'aid': serv_addr['aid'],
        'hash': '62355501e146b801a134fa04367f65b8',
        'rhash': '191a2e2f9fb48ba2d54c35b6f8dae7fc',
        'swfupload': "1",
        'api': "1"
    }
    print('Posting a photo...')

    request = requests.post(url, data, files=files)
    print(request.text)
    request_dict = json.loads(request.text)
    if request.status_code == 200 and request.reason == 'OK':
        api.photos.save(
            server=request_dict['server'],
            photos_list=request_dict['photos_list'],
            aid=request_dict['aid'],
            hash=request_dict['hash']
        )
    else:
        print('Cannot make a post request')
    sleep(2)


post_photo('album_id_as_an_integer', photo='test.png')
