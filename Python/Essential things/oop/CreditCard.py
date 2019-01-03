class CreditCard(object):
    """A consumer credit card."""

    def __init__(self, customer, bank, account, limit):
        """Create a new credit card instance.
        
        The initial balance is zero.

        customer:    the name of the customer, String
        bank:        the name of the bank, String
        account:     the account identifier, String of digits
        limit:       credit limit, in dollars, Integer
        """
        self._customer = customer
        self._bank = bank
        self._account = account
        self._limit = limit
        self._balance = 0

    def get_customer(self):
        """Return name of the customer."""
        return self._customer

    def get_bank(self):
        """Return the bank's name."""
        return self._bank

    def get_account(self):
        """Return the card identifying number."""
        return self._account

    def get_limit(self):
        """Return current credit limit."""
        return self._limit

    def get_balance(self):
        """Return current balance."""
        return self._balance

    def charge(self, price):
        """Charge given price to the card, assuming sufficient credit limit.
        
        Returns: 
            True if charge was processed;
            False if charge was denied.
        """
        if (self._balance + price) > self._limit:
            return False  # if charge would exceed limit, cannot accept charge.
        else:
            self._balance += price
            return True

    def make_payment(self, amount):
        """Process customer payment that reduces balance."""
        self._balance -= amount

if __name__ == "__main__":
    wallet = []
    wallet.append(CreditCard('a', 'b', '1 2 3 4', 1000))
    wallet.append(CreditCard('c', 'd', '2 3 4 5', 2000))
    wallet.append(CreditCard('e', 'f', '3 4 5 6', 3000))

    for value in range(10):
        wallet[0].charge(value)
        wallet[1].charge(2 * value)
        wallet[2].charge(3 * value)
    
    for cc in wallet:
        print('Customer: {0}'.format(cc.get_customer()))
        print('Bank: {0}'.format(cc.get_bank()))
        print('Account: {0}'.format(cc.get_account()))
        print('Limit: {0}'.format(cc.get_limit()))
        print('Balance: {0}'.format(cc.get_balance()))
        while cc.get_balance() > 100:
            cc.make_payment(100)
            print('New balance: {0}'.format(cc.get_balance()))
        print('---\n')
