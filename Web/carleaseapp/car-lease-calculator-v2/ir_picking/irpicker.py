class IRDetailsHandler(object):

    def __init__(
        self,
        loan_details: dict,
        income_details: dict,
        expenses_details: dict
    ):
        self.loan_details = loan_details
        self.income_details = income_details
        self.expenses_details = expenses_details
        self.ir_threshold = 0.7

    def handle(self) -> dict:
        total_income = self._evaluate_total_income()
        total_expenses = self._evaluate_total_expenses()
        ir = self._evaluate_ir(total_income, total_expenses)
        decision = self._evaluate_decision(ir)
        suggestion = self._evaluate_suggestion(ir)
        loan_decrease = self._evaluate_loan_decrease(total_income, total_expenses)
        # Prepare storage with results
        result = {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'ir': ir,
            'decision': decision,
            'suggestion': suggestion,
            'loan_decrease': loan_decrease,

            'ir_threshold': self.ir_threshold
        }
        return result

    def _evaluate_total_income(self) -> float:
        # Evaluate client total income part
        client_income_details = self.income_details.get('client', {})
        client_income = \
            client_income_details.get('confirmed', None) + \
            client_income_details.get('unconfirmed', None) * client_income_details.get('unconfirmed_type_coef', None)
        # Evaluate spouse total income part
        spouse_income_details = self.income_details.get('spouse', {})
        spouse_income_details_empty = \
            spouse_income_details.get('total', None) is None or \
            spouse_income_details.get('confirmed', None) is None or \
            spouse_income_details.get('unconfirmed', None) is None
        if spouse_income_details_empty:
            spouse_income = 0.0
        else:
            spouse_income = \
                spouse_income_details.get('confirmed', None) + \
                spouse_income_details.get('unconfirmed', None) * spouse_income_details.get('unconfirmed_type_coef', None)
        # Evaluate total income
        total_income = client_income + spouse_income
        return total_income

    def _evaluate_total_expenses(self) -> float:
        total_expenses = \
            self.expenses_details.get('rent', None) + \
            self.expenses_details.get('existing_loans', None) + \
            self.expenses_details.get('family_members_count', None) * self.expenses_details.get('living_city_family_member_expenses', None)
        return total_expenses

    def _evaluate_ir(
        self,
        total_income: float,
        total_expenses: float
    ) -> float:
        loan_payment = self.loan_details.get('monthly_payment', None)
        ir = loan_payment / (total_income - total_expenses)
        return ir

    def _evaluate_decision(
        self,
        ir: float
    ) -> str:
        if ir > self.ir_threshold:
            return 'Reject'
        else:
            return 'Approve'

    def _evaluate_suggestion(
        self,
        ir: float
    ) -> list:
        # Suggest OK message if IR is sufficient
        if ir < self.ir_threshold:
            suggestion = ['OK']
            return suggestion
        # Suggest actions if IR is not sufficient
        spouse_income_details = self.income_details.get('spouse', {})
        spouse_income_details_empty = \
            spouse_income_details.get('total', None) is None or \
            spouse_income_details.get('confirmed', None) is None or \
            spouse_income_details.get('unconfirmed', None) is None
        if spouse_income_details_empty:
            suggestion = ['Add Spouse', 'Decrease Loan Amount']
        else:
            suggestion = ['Decrease Loan Amount']
        return suggestion

    def _evaluate_loan_decrease(
        self,
        total_income: float,
        total_expenses: float
    ) -> float:
        loan_amount_step = 1000.0
        base_loan_amount = self.loan_details.get('amount', None)
        while base_loan_amount > 0:
            # Evaluate IR for current loan amount
            current_ir = self._evaluate_ir(total_income, total_expenses)
            # Check whether current loan amount is sufficient for IR threshold
            if current_ir < self.ir_threshold:
                return base_loan_amount
            # Iterate to the next loan amount
            base_loan_amount -= loan_amount_step
        
        # If none of loan amount is enough to fullfill IR, return 0.0
        return 0
