import os
from yattag import Doc
from yattag import indent
# tag(): add two tags: opening and closing.
# doc.stag(): add self closing tag.
# doc.attr - sets the value(s) of one or more attributes of the current tag.
# doc.input - create an input tag. You can use [defaults] with it.
# doc.textarea - create a textarea tag. You can use [defaults] with it.
# [defaults] in Doc(...) - autocompletion of forms.
# [errors] in Doc(...) - creates a <span class="error">_your_text_</span>


class HtmlCreator:
    def __init__(self):
        self.cur_path = os.getcwd()
        # Create header, body and write the whole content to html file.
        self.initDocTagText()

    def initDocTagText(self):
        """
        Create main yattag variables.
        This method can be overridden.
        """
        self.doc, self.tag, self.text = Doc().tagtext()

    def create_html(self, html_name='test.html'):
        """
        Create a html file from header and body content.
        Next, write html content to the hard drive.
        This method cannot be overridden.
        """
        # Add html content to the self.doc
        self.doc.asis('<!DOCTYPE html>')
        with self.tag('html'):
            self.design_header()
            self.design_body()
        # Write html content from self.doc
        with open(html_name, 'w') as f:
            html_content = indent(
                self.doc.getvalue(),
                indentation='  ',
                newline='\n'
            )
            f.write(html_content)

    def design_header(self):
        """
        Create a header for your html file here.
        This method should be overridden.
        """
        pass

    def design_body(self):
        """
        Create a body for your html file here.
        This method should be overridden.
        """
        pass


class MyHtml(HtmlCreator):
    def initDocTagText(self):
        self.doc, self.tag, self.text = Doc(
            defaults={
                'title': 'Untitled',
                'contact_message': 'You just won the lottery!'
            }
        ).tagtext()

    def design_header(self):
        with self.tag('head'):
            with self.tag('title'):
                self.text('A title.')

    def design_body(self):
        with self.tag('body'):
            with self.tag('h1'):
                self.text('Contact form')
            with self.tag('form', action=""):
                self.doc.input(name='title', type='text')
                with self.doc.textarea(name='contact_message'):
                    pass
                self.doc.stag('input', type='submit', value='Send my message')


if __name__ == '__main__':
    myhtmlobj = MyHtml()
    myhtmlobj.create_html()
