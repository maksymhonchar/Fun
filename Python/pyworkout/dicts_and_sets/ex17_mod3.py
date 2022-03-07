import os


def get_extensions_stats(
    folder_path: str
) -> set:
    extension_idx = 1
    stats = {
        os.path.splitext(filename)[extension_idx]
        for filename in os.listdir(folder_path)
    }
    return stats


def main():
    folder_path = r'...'
    stats = get_extensions_stats(folder_path)
    print(stats)


if __name__ == '__main__':
    main()
