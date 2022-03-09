import pprint
import os


def analyze_folder_files(
    folder_path: str
) -> dict:
    analysis = {}

    try:
        for filename in os.listdir(folder_path):
            filename_path = os.path.join(folder_path, filename)
            analysis[filename_path] = os.stat(filename_path).st_size
    except FileNotFoundError:
        error_msg = f'analyze_folder_files failed: cant find [{folder_path}]'
        raise FileNotFoundError(error_msg)

    return analysis


def main():
    folder_path = '.'
    analysis = analyze_folder_files(folder_path)
    pprint.pprint(analysis)


if __name__ == '__main__':
    main()
