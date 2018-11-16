from django.test import TestCase

from rdmsapp.models import Contact

class ContactTests(TestCase):
    """Contact model tests."""
    def test_str(self):
        contact = Contact(
            first_name='a',
            last_name='b',
        )
        self.assertEquals(str(contact), 'a b')
