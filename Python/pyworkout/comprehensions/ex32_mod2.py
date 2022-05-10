import os


def analyze_dir(
    dir_path: str
) -> dict:
    return {
        filename: os.path.getsize(os.path.join(dir_path, filename))
        for filename in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, filename))
    }


def main():
    dir_path = './comprehensions/'
    result = analyze_dir(dir_path)
    print(result)


if __name__ == '__main__':
    main()
