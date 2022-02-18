import numpy_financial as npf


def evaluate_irr(
    cashflow: list
) -> float:
    """
    Evaluate Internal Rate of Return (IRR) for given cashflow.

    Docs:
        https://numpy.org/numpy-financial/latest/irr.html#numpy_financial.irr

    Args:
        cashflow: list - list of floats which represent Input cash flows per time period

    Returns:
        float: Internal Rate of Return for periodic input values
    """
    irr = npf.irr(values=cashflow)
    return irr


def evaluate_pmt(
    rate: float,
    n_periods: int,
    present_value: float,
    future_value=0
) -> float:
    """
    Evaluate payment against loan principal plus interest.

    Docs:
        https://numpy.org/numpy-financial/latest/pmt.html#numpy_financial.pmt
        The payment is computed by solving the equation:
            fv +
            pv*(1 + rate)**nper +
            pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0

    Args:
        rate: float - rate of interest per period
        n_periods: int - number of compounding periods
        present_value: float - present value, usually price-upfront_payment
        future_value: float - future value. Usually 0

    Returns:
        float: payment against loan plus interest
    """
    return npf.pmt(rate=rate, nper=n_periods, pv=present_value, fv=future_value)


def find_rate_having_irr(
    customer_metrics: dict,
    business_metrics: dict,
    precision=1
) -> dict:
    """
    Find rate, having customer and business metrics, so that IRR equals to specified target value.

    Note:
        This is an exhausting algorithm that simply computes lots of IRR values.

    Args:
        customer_metrics: dict - set of 3 customer metrics: object price, upfront payment, lease term
        business_metrics: dict - set of 2 business metrics: commission rate, desired IRR value
        precision: int - digits after comma in found rate value

    Returns:
        tuple: (dict, dict) - first element describes all candidates, second element describes best candidate
    """
    # Find out customer and business metrics
    price, upfront_payment, term = customer_metrics['price'], customer_metrics['upfront_payment'], customer_metrics['term']
    commission_rate, target_irr = business_metrics['commission_rate'], business_metrics['target_irr']
    # Find rates for all candidates
    candidates = []
    investment = (price - upfront_payment - price*commission_rate) * -1.0
    for candidate_rate in range(0, 100*10**precision):
        # Evaluate IRR for the rate candidate
        candidate_rate_pct = candidate_rate / (100*10**precision)
        candidate_monthly_rate_pct = candidate_rate_pct / 12.0
        monthly_payment = evaluate_pmt(
            rate=candidate_monthly_rate_pct,
            n_periods=term,
            present_value=price-upfront_payment
        ) * -1.0
        candidate_irr = evaluate_irr(
            [investment] + [monthly_payment] * term
        ) * 12
        # Evaluate absolute difference between target IRR and candidate IRR
        target_candidate_irr_diff_abs = abs(target_irr - candidate_irr)
        # Save candidate characteristics
        candidates.append(
            {
                'rate': candidate_rate_pct,
                'monthly_rate': candidate_monthly_rate_pct,
                'investment': investment,
                'monthly_payment': monthly_payment,
                'irr': candidate_irr,
                'target_candidate_irr_diff_abs': target_candidate_irr_diff_abs
            }
        )
    # Find best candidate
    candidates.sort(
        key=lambda characteristics: characteristics['target_candidate_irr_diff_abs']
    )
    best_candidate = candidates[0]

    return candidates, best_candidate
