import vk
import datetime
import time
import re
import urllib.request as urllib2
import http.cookiejar as cookielib
import html.parser as parser
import urllib.parse as urllib


# id of my app in vk: [5521254]
session = vk.Session()
api = vk.API(session, v='5.53', lang='ru', timeout=10)


def get_servtime():
    def unix_to_normal(unix_timestamp):
        return datetime.datetime.fromtimestamp(
            int(unix_timestamp)
        ).strftime('%Y-%m-%d %H:%M:%S')

    servtime = api.getServerTime()
    current_time = time.time()
    difference = int(unix_to_normal(servtime)[-2:]) - int(unix_to_normal(current_time)[-2:])
    return '%i) seconds' % difference


class FormParser(parser.HTMLParser):
    def __init__(self):
        parser.HTMLParser.__init__(self)
        self.url = None
        self.params = {}
        self.in_form = False
        self.form_parsed = False
        self.method = "GET"

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "form":
            if self.form_parsed:
                raise RuntimeError("Second form on page")
            if self.in_form:
                raise RuntimeError("Form tag in form")
            self.in_form = True
        if not self.in_form:
            # Handle other tags.
            return
        attrs = dict(
            (name.lower(), value) for name, value in attrs
        )
        if tag == "form":
            self.url = attrs["action"]
            if "method" not in attrs:
                self.method = attrs["method"]
        elif tag == "input" and "type" in attrs and "name" in attrs:
            if attrs["type"] in ["hidden", "text", "password"]:
                if "value" in attrs:
                    self.params[attrs["name"]] = attrs["value"]
                else:
                    self.params[attrs["name"]] = ""

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "form":
            if not self.in_form:
                raise RuntimeError("Unexpected end of form tag")
            self.in_form = False
            self.form_parsed = True


def build_token(scope=[], state='default', mail='123', password='123'):
    url = '&'.join(
        (
            'https://oauth.vk.com/authorize?client_id=5521254',
            'display=page',
            'redirect_uri=https://oauth.vk.com/blank.html',
            'scope=%s' % ','.join(scope),
            'response_type=token',
            'v=5.53',
            'state=%s' % state
        ),
    )
    opener = urllib2.build_opener(
        urllib2.HTTPCookieProcessor(cookielib.CookieJar()),
        urllib2.HTTPRedirectHandler()
    )
    response = opener.open(url)
    vkform_parser = FormParser()
    vkform_parser.feed(response.read().decode("utf-8"))
    vkform_parser.params["email"] = mail
    vkform_parser.params["pass"] = password
    data_to_post = urllib.urlencode(vkform_parser.params)
    binary_data_to_post = data_to_post.encode('utf-8')
    print('Starting to get a response...')
    response = opener.open(
        vkform_parser.url,
        binary_data_to_post
    )
    print('Got the token!')
    vk_token = re.search(r'(.*)access_token=(.*)&expires_in(.*)', response.url)
    return vk_token.group(2)


token = build_token(
        scope=['scope_params'],
        mail='login',
        password='pass'
    )
print(token)
