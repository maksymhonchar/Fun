import os

def disk_usage(path):
    """Return the number of bytes used by a file/folder and any descendents."""
    total = os.path.getsize(path)
    if os.path.isdir(path):
        for filename in os.listdir(path):
            childpath = os.path.join(path, filename)
            print('Exploring {0}'.format(childpath))
            total += disk_usage(childpath)
    print('{0:<7} mb in {1}'.format(total / 1024 / 1024, path))
    return total

if __name__ == "__main__":
    disk_usage('C:\\Users\\maxgo\\Desktop\\sw')
