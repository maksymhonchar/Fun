import hashlib
import os


def get_md5(
    content: str
) -> str:
    return hashlib.md5( content.encode('utf-8') ).hexdigest()


def get_hashes_for_dir_files(
    dir_path: str
) -> dict:
    hashes = {}

    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath):
            with open(filepath) as fs_r:
                hashes[filename] = get_md5( fs_r.read() )

    return hashes


def main():
    dir_path = '.'
    hashes = get_hashes_for_dir_files(dir_path)
    print(hashes)


if __name__ == '__main__':
    main()
