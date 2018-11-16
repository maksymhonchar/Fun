import vk
from time import sleep


class User(object):

    likesCount = 0

    def __init__(self, uid):
        self.id = uid
        self.api = self.auth()

    @staticmethod
    def auth():
        # Auth key is the fastest approach.
        # To authenticate from vk app, use following:
        # session = vk.AuthSession(app_id='appid', user_login='jake@gmail.com', user_password='Finn')
        session = vk.Session(access_token='YourAmazingToken')
        return vk.API(session)

    def addLikes_post(self, owner_id, post_id, offset):
        post_comments = self.api.wall.getComments(
            owner_id=int(owner_id),
            post_id=int(post_id),
            need_likes=1,
            offset=offset,
            count=100
        )
        post_commentsAmount = post_comments[0]
        for i in range(post_commentsAmount):
            if post_comments[i + 1]['likes']['can_like'] == 0:
                continue
            response = self.api.likes.add(
                type='comment',
                owner_id=int(owner_id),
                item_id=post_comments[i + 1]['cid']
            )
            sleep(5)
            User.likesCount += 1
            print('Response from certain comment', response)
            print(User.likesCount, str(owner_id)+'_'+str(post_comments[i + 1]['cid']))
        sleep(5)

    def addLikes_wall(self, domain):
        posts = self.api.wall.get(domain=domain, count=100)
        posts_amount = posts[0]
        offset = 0
        for i in range(posts_amount):
            if i+1 % 100 == 0:  # 101 ?
                offset += 100
            owner_id = posts[i + 1]['to_id']
            post_id = posts[i + 1]['id']
            self.addLikes_post(owner_id, post_id, offset)


def main():
    user_id = 'userid_digits_only'
    vkUser = User(user_id)

    # To add likes to certain post, use this method.
    # Offset should be 0, if you use this method manually.
    
    # vkUser.addLikes_post(targetid, post_id, offset)

    # To add likes to the whole wall, use next method.
    # Param: [owner_id] - domain name of target.
    # Examples: 'overhearkpi', 'id291823738' 'diferenzial13'
    try:
        vkUser.addLikes_wall('overhearkpi')
    except vk.exceptions.VkAPIError as e:
        print('VkAPIError: {0}'.format(e))

    print('End of the program')


if __name__ == '__main__':
    main()
