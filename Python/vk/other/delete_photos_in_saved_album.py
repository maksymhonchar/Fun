import vk
from time import sleep

class User(object):
    def __init__(self, uid):
        self.id = uid
        self.api = self.auth()

    @staticmethod
    def auth():
        session = vk.AuthSession(access_token='123')
        return vk.API(session, v='5.63')

    def test(self):
        # print(self.api.users.get(user_ids=1))
        print(self.api.getServerTime())

    def del_photos(self):
        album_photos_list = self.api.photos.get(owner_id=self.id, album_id="saved", count=1000)
        for photo in album_photos_list['items']:
            photo_id = photo['id']
            self.api.photos.delete(owner_id=self.id, photo_id=photo_id)
            print('Photo with id {0} deleted.'.format(photo_id))
            sleep(1)


def main():
    # Initial data (I am too lazy to do the whole validation routine in console).
    user_id = 123
    vkUser = User(user_id)
    # Delete everything in your [saved] album.
    try:
        vkUser.del_photos()
    except vk.exceptions.VkAPIError as e:
        print('VkAPIError: {0}'.format(e))
    # Show end of the program.
    print('End of the program.')


if __name__ == '__main__':
    main()
