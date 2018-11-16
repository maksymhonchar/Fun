# 1. reading and writing csv data.
import csv

def read_csv_as_tuples():
    with open('stocks.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        print('headers:', headers)
        print(type(headers), type(f_csv))
        for i, row in enumerate(f_csv):
            print(str(i+1) + '.', headers[i], row)
    print('end of "csv" file.')

from collections import namedtuple
def read_csv_as_tuples_2():
    with open('stocks.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        Row = namedtuple('Row', headers)
        for r in f_csv:
            row = Row(*r)
            print(row)
            for item in row:
                print(type(item), end=' ')
            try:
                print('\n', row.Symbol_miss)
            except AttributeError:
                print('\nmissing')

def read_csv_as_dicts():
    with open('stocks.csv') as f:
        f_csv = csv.DictReader(f)
        rows = {}
        for row in f_csv:
            rows[row.values()] = row
            print(row.values(), type(row.values()))
            print(row)
            try:
                print(row['Symbol'])
            except KeyError:
                print('missing')
        # print(next(f_csv))
        print('\n\n', rows)

def write_csv():
    headers = [1, 2, 3, 4, 5]
    rows = [ ('a', 'b', 'c', 'd', 'e'), ('a', 'b', 'c', 'd', 'e') ]
    with open('stocks2.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)

    rows_dicts = [ {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, {1: 6, 2: 7, 3: 12, 4: 9, 5: 8} ]
    with open('stocks3.csv', 'w') as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        f_csv.writerows(rows_dicts)

def read_csv_delimeter():
    import requests
    with open('stocks_tab.tsv', newline='\n') as f:
        reader = csv.reader(f, csv.excel_tab)
        for row in reader:
            print(row)
    print('__')
    # below - the same
    req = requests.get('http://vote.wa.gov/results/current/export/MediaResults.txt')
    req_data = req.text
    reader = csv.reader(req_data.splitlines(), csv.excel_tab)
    for row in reader:
        print(row)

def read_csv_broken_named_tuples():
    import re
    with open('stocks.csv') as f:
        f_csv = csv.reader(f)
        # next line: "some-?!?value" => "some____value"
        headers = [ re.sub('[^a-zA-Z_]', '_', h) for h in next(f_csv) ]
        print(headers)
        Row = namedtuple('Row', headers)
        for r in f_csv:
            cur_row = Row(*r)
            print(cur_row)

def read_csv_strict_type():
    types = [
        ('Price', float),
        ('Change', float),
        ('Volume', int)
    ]
    with open('stocks.csv') as f:
        for row in csv.DictReader(f):
            row.update(
                (key, conversion(row[key]))
                for key, conversion in types
            )
            print(type(row['Change'])) # <class 'float'>
            print(row)

def read_csv_empty_values():
    # [1] - price -> float
    with open('stocks.csv') as f:
        rdr = csv.reader(f)
        next(rdr) # skip the headings
        for row in rdr:
            cur_price = float(row[1]) if row[1] != '' else 0
            print(cur_price, type(cur_price))

# 2. reading and writing json data.
import json

def read_write_json():
    # dumps and loads - for builtins
    data = {
        'name': 'maxim',
        'age': 18,
        'height': 1.0
    }
    json_str = json.dumps(data, indent=True)
    print(json_str)
    data_back = json.loads(json_str)
    print(data_back)
    # dump and load - for files.
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=True)
    with open('data.json', 'r') as f:
        data_back = json.load(f)
    print(data_back)

    test_dumps = json.dumps(False)
    print(test_dumps, type(test_dumps)) #str
    test_dumps = json.dumps(
        {
            False : True
        }
    )
    print(test_dumps, type(test_dumps)) # {"false" : true}, false -> str

def json_pretty_alphab_sort():
    from urllib.request import urlopen
    from pprint import pprint
    
    u = urlopen('http://search.twitter.com/search.json?q=python&rpp=5')
    resp = json.loads(u.read().decode('utf-8'))
    pprint(resp)

def json_read__hook_ordered_dict():
    s = '{"name": "maxim","age": 18,"height":1.0}'
    from collections import OrderedDict
    data = json.loads(s, object_pairs_hook=OrderedDict)
    print(data)

    class JSONObject:
        def __init__(self, d):
            self.__dict__ = d
    data = json.loads(s, object_hook=JSONObject)
    print(data.name, data.age, data.height)

def json_misc():
    data = {
        'name': 'maxim',
        'age': 18,
        'height': 1.0
    }
    print(json.dumps(data, indent=10))
    print(json.dumps(data, sort_keys=True, indent=2))

def json_serialize():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    def serialize_instance(obj):
        d = { '__classname__' : type(obj).__name__ }
        d.update(vars(obj))
        return d  # dict!
    p = Point(2, 3)
    serlz_point = serialize_instance(p)
    print(json.dumps(serlz_point, sort_keys=True))

    # Get instance back:
    classes = {
        'Point' : Point
    }
    def unserialize_object(d):
        clsname = d.pop('__classname__', None)
        if clsname:
            cls = classes[clsname]
            obj = cls.__new__(cls)  # make instance without calling __init__
            for key, value in d.items():
                setattr(obj, key, value)
                return obj
        else:
            return d
    deserlz_point = unserialize_object(serlz_point)
    print(vars(deserlz_point), type(deserlz_point))

# 3. Parsing Simple XML Data.
from urllib.request import urlopen
from xml.etree.ElementTree import parse as xmltree_parse, parse


def xml_simple_parse_data():
    # Download rss feed and parse it.
    u = urlopen('http://planetpython.org/rss20.xml')
    doc = xmltree_parse(u)

    doc_root = doc.getroot()
    print(doc_root, type(doc_root))
    print(doc_root.tag, doc_root.text)

    # extract and output tags.
    for item in doc.iterfind('channel/item'):
        title = item.findtext('title')
        date = item.findtext('pubDate')
        link = item.findtext('link')
        print(title, date, link, sep='\n')
        print(type(title), type(date), type(link)) # everything => class <str>
        break

    elem = doc.find('channel/title')
    print("__", elem, elem.tag, elem.text)
    print(elem.get('some_attr'))

from xml.etree.ElementTree import iterparse as xml_iterparse
def parse_and_remove(filename, path):
    # using iterators and generators
    path_parts = path.split('/')
    doc = xml_iterparse(filename, ('start', 'end'))
    # skip the root element
    next(doc)

    tag_stack = []
    elem_stack = []
    for event, elem in doc:
        if event == 'start':
            tag_stack.append(elem.tag)
            elem_stack.append(elem)
        elif event == 'end':
            if tag_stack == path_parts:
                yield elem
                elem_stack[-2].remove(elem)
            try:
                tag_stack.pop()
                elem_stack.pop()
            except IndexError:
                pass

from xml.etree.ElementTree import parse as xml_parse
from collections import Counter
def hugexml_bad_memory_parse():
    doc = xml_parse('items_bigsize.xml')
    items_by_zip = Counter()

    for item in doc.iterfind('row/row'):
        items_by_zip[item.findtext('zip')] +=1

    for zipcode, num in items_by_zip.most_common():
        print(zipcode, num)

from collections import Counter
def hugexml_good_memory_parse():
    items_by_zip = Counter()
    data = parse_and_remove('items_bigsize.xml', 'row/row')

    for item in data:
        items_by_zip[item.findtext('zip')] += 1

    for zipcode, num in items_by_zip.most_common():
        print(zipcode, num)

from xml.etree.ElementTree import Element as XMLElement
def xml_to_dictionary(tag, d):
    '''
    Turn a simle kv dict into XML
    '''
    elem = XMLElement(tag)
    for key, val in d.items():
        child = XMLElement(key)
        child.text = str(val)
        elem.append(child)

    # attach to root attribute:
    elem.set('_id', '1234')
    return elem

def print_xml_element_instance(elem):
    from xml.etree.ElementTree import tostring
    print(tostring(elem))

def rough_dict_to_xml_str(tag, d):
    '''
    Turn a kv dict pairs into xml bad workaround
    '''
    parts = ['<{}>'.format(tag)]
    for key, val in d.items():
        parts.append('<{0}>{1}<{0}>'.format(key, val))
    parts.append('</{}>'.format(tag))
    print(parts)

    #
    # A problem here, if dictionary d contains items like:
    # { 'name': '<the_tag>' }
    # workaround: in def proper_xml_escape()
    #

    return ''.join(parts)

from xml.etree.ElementTree import Element as XMLElement
from xml.sax.saxutils import escape, unescape
def proper_manual_xml_escape(tag, d):
    root = XMLElement(tag)
    for key, val in d.items():
        child = XMLElement(key)
        child.text = escape(str(val))
        root.append(child)
    # for test:
    __t = '<the_unescaped_tag>'
    print('unescaped:', unescape(__t))
    print('escaped:', escape(__t))  # out: <class str: 'escaped: &lt;the_unescaped_tag&gt;'>
    return root

# 4. Modifying and rewriting and others with XML

from xml.etree.ElementTree import parse, Element as XMLElement
from xml.etree.ElementTree import tostring
def change_xml():
    doc = parse('testingxml.xml')
    root = doc.getroot()
    print(root)

    # remove
    root.remove(root.find('sri'))

    # insert
    root.getchildren().index(root.find('nm'))
    e = XMLElement('newelem')
    e.text = 'This is a new element'
    root.insert(2, e)

    root.getchildren().index(root.find('pre'))
    e = XMLElement('sri')
    e.text = 'a new new text'
    root.insert(5, e)

    # write to a file
    doc.write('testingxml.xml', xml_declaration=True)

def manual_search_xml():
    doc = parse('xmltest.xml')
    root = doc.getroot()

    p = print
    p(doc.findtext('author'))
    p(doc.find('content'))
    doc.find('content/html')
    doc.find('content/{http://www.w3.org/1999/xhtml}html')  # attribute

class XMLNamespaces:
    def __init__(self, **kwargs):
        self.namespaces = []
        for name, uri in kwargs.items():
            self.register(name, uri)
    def register(self, name, uri):
        self.namespaces[name] = '{'+uri+'}'
    def __call__(self, path):
        return path.format_map(self.namespaces)

def search_xml():
    doc = parse('xmltest.xml')
    root = doc.getroot()

    ns = XMLNamespaces(html='http://www.w3.org/1999/xhtml')
    doc.find(ns('content/{html}html'))


def main():
    search_xml()

if __name__ == '__main__':
    main()
