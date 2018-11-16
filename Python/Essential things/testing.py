import unittest


class TestUM(unittest.TestCase):
    def setUp(self):
        """This method executes BEFORE each test"""
        pass

    def tearDown(self):
        """This method executes AFTER each test"""
        pass

    """
    def setUpClass(cls):
        # This method executes BEFORE ALL tests
        print('Testing begins.')

    def tearDownClass(cls):
        # This method executes AFTER ALL tests
        print('Testing complete.')
    """

    def test_numbers_3_4(self):
        self.assertEqual(3*4, 12)

    def test_strings_a_3(self):
        self.assertEqual('a'*3, 'aaa')


"""
List of different checks:
testAssertTrue   |  Invoke error, if argument != True
testFailUnless   |  (Outdated) Invoke error, if argument != True

testAssertFalse  |  Invoke error, if argument != False
testFailIf       |  (Outdated) Invoke error, if argument != False

testEqual        |  Check if two arguments are equal.
testEqualFail    |  (Outdated) Invoke error, if arguments are equal

testNotEqual     |  Check if two arguments aren't equal
testNotEqualFail |  (Outdated) Invoke error, if arguments aren't equal

assertNotAlmostEqual  |  Compare two arguments with rounding. Invoke error if arguments are equal.
testNotAlmostEqual    |  (Outdated) Same as assertNotAlmostEqual

assertAlmostEqual |  Compare two arguments with rounding. Invoke error if arguments aren't equal.
testAlmostEqual   |  (Outdated) Same as assertAlmostEqual
"""

if __name__ == '__main__':
    unittest.main()

# Pretty interesting: http://www.drdobbs.com/testing/unit-testing-with-python/240165163
