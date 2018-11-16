import vk

session = vk.Session(
    access_token='amaaazing token!!')
api = vk.API(session)

kostyaid = 316728306

answer = api.users.get(user_ids=268281914)
api.wall.post(message='Privetiki!', owner_id='kostyaid', )
api.messages.send(user_id='kostyaid', message='hello from python, kostya!')

api.wall.createComment(owner_id='kostyaid', post_id='12', message='koroche, vk with python - ochen-ochen prosto)')
api.wall.createComment(owner_id='kostyaid', post_id='12', message='Comment above created from .py script')
api.wall.createComment(owner_id='kostyaid', post_id='12', message='Comment above too')

# print(type(answer[0]), answer)
