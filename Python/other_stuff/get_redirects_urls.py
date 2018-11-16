import requests

def get_redirects_path(url):
    """Not for oauth :("""
    response = requests.get(url, allow_redirects=True)
    if response.history:
        print('Request was redirected.')
        for resp in response.history:
            print(resp.status_code, resp.url)
        print('Final destination:')
        print(response.status_code, response.url)
    else:
        print('Request was not redirected.')
        print('Current destination:', response.status_code, response.url)
