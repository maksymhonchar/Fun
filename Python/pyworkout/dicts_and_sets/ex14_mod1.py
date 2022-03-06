import hashlib
import os


class User(object):

    def __init__(
        self,
        username,
        password
    ) -> None:
        self.salt = os.urandom(64)
        self.username = username
        self.password = self.encrypt_password(password, self.salt)

    @staticmethod
    def encrypt_password(
        password: str,
        salt: bytes
    ) -> bytes:
        encrypted_password = hashlib.pbkdf2_hmac(
            hash_name='sha256',
            password=password.encode('utf-8'),
            salt=salt,
            iterations=10**5,
            dklen=32
        )
        return encrypted_password

    def __repr__(self) -> str:
        return f'username: [{self.username}]; password: [{self.password}]'


def login(
    users: dict
) -> None:
    while True:
        user_input_username = input('Username: ')

        user_input_username_empty = user_input_username == ''
        if user_input_username_empty:
            print('Aborting login routine')
            return

        if user_input_username in users:
            break
        else:
            print(f'User [{user_input_username}] is missing. Try again')
            continue

    user = users[user_input_username]

    while True:
        user_input_password = input(f'[{user.username}] Password: ')

        user_input_password_empty = user_input_password == ''
        if user_input_password_empty:
            print('Aborting login routine')
            return

        user_salt = user.salt
        user_password = user.password
        password_to_compare = User.encrypt_password(user_input_password, user_salt)

        if user_password == password_to_compare:
            print('Successful login!')
            break
        else:
            print('Incorrect password, try again')
            continue


def main():
    users = {}
    usernames = ('max', 'alex', 'mikhail', 'evhen', 'dima')
    for username in usernames:
        dummy_password = f'{username}123'
        users[username] = User(username=username, password=dummy_password)

    login(users)


if __name__ == '__main__':
    main()
