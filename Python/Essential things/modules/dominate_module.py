import dominate
from dominate.tags import *


def writefile(content='', path=''):
    with open(path, 'w') as f:
        f.write(content)


# Main example below.
# Basic var to contain html content.
doc = dominate.document(title='My tag')

# Manage a <head> block.
with doc.head:
    link(rel='stylesheet', href='style.css')
    script(type='text/javascript', src='srcipt.js')

# Manage a <body> block.
with doc:
    # Create an ordered list
    with div(id='header').add(ol()):
        for i in ['first', 'second', 'third']:
            li(a(i.title(), href='/%s.html' % i))

    with div():
        # Add attributes inside of [with] manager
        attr(cls='body')
        # [<] or [>] or other characters are safe.
        p('My own <p> tag, crazy')


def feature():
    content = html(
        body(
            h1('H1 tag.')
        )
    )
    print(content)


def feature2():
    # For custom HTML5 data attributes.
    content = div(
        data_myAmazingName='101011'
    )
    print(content)


def feature3():
    # Possible to modify attributes as a dict.
    header = div()
    header['id'] = 'header'
    print(header)  # <div id="header"></div>


def feature4():
    # Lists with [+=] and [.add()]
    list = ul()
    print(type(list))  # <class 'dominate.tags.ul'>
    for item in range(5):
        list += li('item #', item)
    list.add(
        li('another item, hmm...')
    )
    print(list)


def feature5():
    # Iterators!
    menu_items = [
        ('google', 'google.com'),
        ('yandex', 'yandex.ru'),
        ('vk', 'vk.com')
    ]
    unordered_list = ul(
        li(
            a(
                name, href=link
            ),
            __pretty=False
        )
        for name, link in menu_items
    )
    print(unordered_list)  # beautiful list with links!
    print(unordered_list[0])  # print(unordered_list[0])


def feature6():
    # A simple document tree.
    _html = html()
    _head = _html.add(head(title('A title!')))
    _body = _html.add(body())
    hdr = _body.add(div(id='hdr'))
    content = _body.add(div(id='content'))
    footer = _body.add(div(id='footer'))
    print(_html)  # indented html file

    print('----------------------')

    # Or create it with a clean-up code style
    _html = html()
    _head, _body = _html.add(head(title('A title!')), body())
    blocks = ['hdr', 'content', 'footer']
    header, content, footer = _body.add([div(id=name) for name in blocks])
    print(_html)
    print(header, content, footer, sep='\n--------\n')

    # Comment
    print(comment('I am a html comment'))

    print(comment(p('Upgrade to newer IE!'), condition='lt IE9'))  # if-endif in html5


def feature7():
    llist = ul()
    with llist:
        li('one')
        li('two')
        li('three')
    print(llist)

@div
def greetin(name):
    p('hi! My name is %s!' %name)  # div-p-/p-/div
#print(greetin('Maxim'))


def feature8():
    d = dominate.document()  # creates a ready html5 page.
    print(d)
    print(d.head)  # head contents
    print(d.body)  # body contents
    with open('test.html', 'w') as f:
        f.write(d.__str__())  # or str(d)
