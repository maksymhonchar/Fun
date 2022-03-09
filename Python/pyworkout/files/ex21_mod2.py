import datetime
import os


def display_folder_stats() -> None:
    user_dir_path = input('Enter directory path: ')

    try:
        dir_files = os.listdir(user_dir_path)
        dir_last_modified_dt = datetime.datetime.utcfromtimestamp(
            os.stat(user_dir_path).st_mtime
        )
    except FileNotFoundError:
        error_msg = f'display_folder_stats failed: [{user_dir_path}] is missing'
        raise FileNotFoundError(error_msg)

    print(f'Files in directory [{user_dir_path}]: [{dir_files}]')
    print(f'Directory [{user_dir_path}] last modification dt: [{dir_last_modified_dt}]')


def main():
    display_folder_stats()


if __name__ == '__main__':
    main()
