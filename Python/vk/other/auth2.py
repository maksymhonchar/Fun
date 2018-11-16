import vk


session = vk.Session()
api = vk.API(session, v='5.52', lang='ru', timeout=10)


def get_likes(owner_id):
    """Get amount of likes in last 100 posts of any type"""
    try:
        response = api.wall.get(owner_id=owner_id, count=100)
    except vk.exceptions.VkAPIError:
        print('Access denied')
        return
    # Get amount of total users posts.
    postsAmount = response['count']
    if postsAmount >= 100:  # omg
        postsAmount = 99  # omg
    likes = 0
    for i in range(postsAmount):
        # Get amount of likes on the certain post.
        likesStruct = response['items'][i]['likes']
        likes += likesStruct['count']
        if 'user_likes' in likesStruct and likes != 0:
            likes -= 1
    return likes


def get_likes_v2(owner_id):
    """Get amount of likes in last 100 posts of any type"""

    # wall.get request - get response or raise an error.
    try:
        response = api.wall.get(owner_id=owner_id, count=100)
    except vk.exceptions.VkAPIError:
        print('Access denied')
        # TODO: get exception key
        return
    # Amount of total users posts.
    postsAmount = response['count']
    # Get
    lastId = response['items'][0]['id']
    likes = 0
    tmp = 0

    # Do wall.getById request for each id to get each post individually.
    for i in range(lastId + 1):
        # Do wall.getById request and get response.
        try:
            post_id = owner_id + '_' + str(i)
            response = api.wall.getById(posts=post_id)
        except vk.exceptions.VkAPIError:
            print('Access denied')
            # TODO: get exception key
            return
        # Check, if post with certain id exists.
        if len(response) == 0:
            continue
        tmpId = 'vk.com/id' + owner_id + '?w=wall' + str(post_id)
        print('Got it! Id:', tmpId)
        tmp += 1
        # Get likes from certain post!
        likes += response[0]['likes']['count']
        # TODO: -= user_likes if user_likes==1

    # Return total amount of likes on the wall.
    return likes, tmp, postsAmount


def main():
    maximVkId = '291823738'
    durovId = '1'

    # V1
    #likes = get_likes(owner_id=durovId)
    #print(likes)

    # V2
    likes, tmp, postsAmount = get_likes_v2(maximVkId)
    print(likes, tmp, postsAmount)

if __name__ == '__main__':
    main()
