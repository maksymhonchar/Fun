import math
import numpy_financial as npf


class RatePicker(object):

    def pick_v1(
        self,
        customer_metrics: dict,
        business_metrics: dict,
        precision=1
    ) -> dict:
        """Find rate, having customer and business metrics, so that IRR approximates specified target value.

        Note:
            This is an exhausting algorithm that simply computes lots of IRR values.

        Args:
            customer_metrics: dict - set of customer metrics: object price, upfront payment, lease term
            business_metrics: dict - set of business metrics: commission rate, desired IRR value
            precision: int - digits after comma in picked rate value

        Returns:
            dict - best rate approximation
        """
        # Find out customer and business metrics
        price, upfront_payment, term = customer_metrics['price'], customer_metrics['upfront_payment'], customer_metrics['term']
        commission_rate, target_irr = business_metrics['commission_rate'], business_metrics['target_irr']

        # Evaluate initial investment
        investment = (price - upfront_payment - price*commission_rate) * -1.0

        # Evaluate loan rate for each candidate
        candidates = []
        for candidate_rate in range(0, 100*10**precision):
            # Evaluate rate candidate
            candidate_rate_pct = candidate_rate / (100*10**precision)
            # Evaluate PMT
            monthly_payment = npf.pmt(
                rate=candidate_rate_pct / 12.0,
                nper=term,
                pv=price-upfront_payment,
                fv=0
            ) * -1.0
            monthly_payment = math.ceil(monthly_payment)
            # Evaluate IRR
            candidate_irr = npf.irr(
                values=[investment] + [monthly_payment] * term
            ) * 12
            # Evaluate absolute difference between target IRR and candidate IRR
            target_candidate_irr_diff_abs = abs(target_irr - candidate_irr)
            
            # Save candidate characteristics
            candidates.append(
                {
                    'rate': candidate_rate_pct,
                    'investment': investment,
                    'monthly_payment': monthly_payment,
                    'irr': candidate_irr,
                    'target_candidate_irr_diff_abs': target_candidate_irr_diff_abs
                }
            )

        # Find the best candidate
        candidates.sort(
            key=lambda characteristics: characteristics['target_candidate_irr_diff_abs']
        )
        best_candidate = candidates[0]

        return best_candidate

    def pick_v2(
        self,
        customer_metrics: dict,
        business_metrics: dict,
        misc_metrics: dict
    ) -> dict:
        """
        todo: docs
        """
        # Find out customer metrics
        car_price_usd = customer_metrics['car_price_usd']
        down_payment_usd = customer_metrics['down_payment_usd']
        lease_term_months = customer_metrics['lease_term_months']
        discount_pct = customer_metrics['discount_pct']
        # Find out business metrics
        commission_pct = business_metrics['commission_pct']
        tracker_price_uah = business_metrics['tracker_price_uah']
        tracker_subscription_fee_uah = business_metrics['tracker_subscription_fee_uah']
        insurance_pct = business_metrics['insurance_pct']
        desired_irr_pct = business_metrics['desired_irr_pct']
        # Find out other miscellaneous metrics
        exchange_rate = misc_metrics['exchange_rate']
        precision = misc_metrics['precision']

        # Evaluate essential values
        car_price_uah = car_price_usd * exchange_rate
        down_payment_uah = \
            down_payment_usd * exchange_rate - \
            car_price_uah * commission_pct - \
            3000
        investment = \
            ((car_price_uah - down_payment_uah) - car_price_uah * commission_pct) * -1.0 - \
            car_price_uah * insurance_pct - \
            tracker_price_uah
        discount_uah = discount_pct * car_price_uah

        # Evaluate loan rates for all candidates
        candidates = []
        for candidate_rate in range(100 * 10**precision):
            # Evaluate rate candidate
            candidate_rate_pct = candidate_rate / (100 * 10**precision)
            # Evaluate PMT
            pmt_value = npf.pmt(
                rate=candidate_rate_pct / 12.0,
                nper=lease_term_months,
                pv=(car_price_uah-down_payment_uah) * -1.0,
                fv=discount_uah
            )
            monthly_payment = pmt_value + tracker_subscription_fee_uah
            #
            monthly_payments = [monthly_payment] * lease_term_months
            #
            monthly_payments[lease_term_months - 1] += discount_uah
            #
            years_in_lease_term = lease_term_months // 12
            for year_index in range(years_in_lease_term):
                month_index = (year_index + 1) * 12 - 1
                if month_index == lease_term_months - 1:
                    continue
                monthly_payments[month_index] -= car_price_uah * insurance_pct
            # Evaluate IRR
            candidate_irr_pct = npf.irr(
                [investment] + monthly_payments
            ) * 12
            # Evaluate absolute difference between desired IRR and candidate IRR
            target_candidate_irr_diff_abs = abs(desired_irr_pct - candidate_irr_pct)

            # Save candidate characteristics
            candidates.append(
                {
                    'rate': candidate_rate_pct,
                    'investment': investment,
                    'monthly_payment': monthly_payment,
                    'monthly_payments': monthly_payments,
                    'irr': candidate_irr_pct,
                    'target_candidate_irr_diff_abs': target_candidate_irr_diff_abs,
                }
            )
        # Evaluate best candidate
        candidates.sort(
            key=lambda candidate: candidate['target_candidate_irr_diff_abs']
        )
        best_candidate = candidates[0]

        # Evaluate right panel results
        lease_sum_uah = car_price_uah - down_payment_uah
        monthly_payment_uah = best_candidate['monthly_payment']
        overpayment_uah = monthly_payment_uah * lease_term_months - lease_sum_uah
        single_value_evaluations = {
            'lease_sum_uah': lease_sum_uah,
            'monthly_payment_uah': monthly_payment_uah,
            'overpayment_uah': overpayment_uah,
            'discount_uah': discount_uah,
        }

        # Evaluate payment plan results
        payment_schedule = {
            'monthly_payments': [],
            'ppmt_values': [],
            'ipmt_values': [],
            'pmt_values': [],
            'leftover_values': [],
            'leftover_values_chart': []
        }

        current_leftover_value = car_price_uah - down_payment_uah
        payment_schedule['leftover_values_chart'].append(current_leftover_value)

        for month_idx in range(lease_term_months):
            ## monthly payment
            payment_schedule['monthly_payments'].append(
                best_candidate['monthly_payments'][month_idx]
            )
            ## PPMT
            ppmt_value = npf.ppmt(
                best_candidate['rate'] / 12.0,
                month_idx + 1,
                lease_term_months,
                (car_price_uah - down_payment_uah) * -1.0,
                fv=discount_uah
            )
            payment_schedule['ppmt_values'].append(ppmt_value)
            ## IPMT
            ipmt_value = npf.ipmt(
                best_candidate['rate'] / 12.0,
                month_idx + 1,
                lease_term_months,
                (car_price_uah - down_payment_uah) * -1.0,
                fv=discount_uah
            )
            ipmt_value = float(ipmt_value)
            payment_schedule['ipmt_values'].append(ipmt_value)
            ## PMT
            pmt_value = npf.pmt(
                best_candidate['rate'] / 12.0,
                lease_term_months,
                (car_price_uah - down_payment_uah) * -1.0,
                fv=discount_uah
            )
            payment_schedule['pmt_values'].append(pmt_value)
            ## Leftover value
            current_leftover_value -= ppmt_value
            payment_schedule['leftover_values'].append(current_leftover_value)
            if (month_idx != 0) and (month_idx % 12 == 0):  # 12, 24, 36
                payment_schedule['leftover_values_chart'].append(current_leftover_value)
        
        payment_schedule['leftover_values_chart'].append(0.0)

        # Create final data storage
        pick_result = {
            'best_candidate': best_candidate,
            'single_value_evaluations': single_value_evaluations,
            'payment_schedule': payment_schedule
        }

        return pick_result

    def pick_cash_credit(
        self,
        customer_metrics: dict,
        business_metrics: dict,
        misc_metrics: dict
    ) -> dict:
        """
        todo: docs
        """
        # Find out customer metrics
        car_price_uah = customer_metrics['car_price_uah']
        lease_term_months = customer_metrics['lease_term_months']
        # Find out business metrics
        commission_pct = business_metrics['commission_pct']
        desired_irr_pct = business_metrics['desired_irr_pct']
        # Find out other miscellaneous metrics
        precision = misc_metrics['precision']

        # Evaluate essential values
        investment = \
            (car_price_uah * -1) + \
            car_price_uah * commission_pct

        # Evaluate loan rates for all candidates
        candidates = []
        for candidate_rate in range(100 * 10**precision):
            # Evaluate rate candidate
            candidate_rate_pct = candidate_rate / (100 * 10**precision)
            # Evaluate PMT
            pmt_value = npf.pmt(
                rate=candidate_rate_pct / 12.0,
                nper=lease_term_months,
                pv=(car_price_uah + (car_price_uah * commission_pct)) * -1.0,
                fv=0
            )
            monthly_payment = pmt_value
            #
            monthly_payments = [monthly_payment] * lease_term_months
            # Evaluate IRR
            candidate_irr_pct = npf.irr(
                [investment] + monthly_payments
            ) * 12
            # Evaluate absolute difference between desired IRR and candidate IRR
            target_candidate_irr_diff_abs = abs(desired_irr_pct - candidate_irr_pct)

            # Save candidate characteristics
            candidates.append(
                {
                    'rate': candidate_rate_pct,
                    'investment': investment,
                    'monthly_payment': monthly_payment,
                    'monthly_payments': monthly_payments,
                    'irr': candidate_irr_pct,
                    'target_candidate_irr_diff_abs': target_candidate_irr_diff_abs,
                }
            )
        # Evaluate best candidate
        candidates.sort(
            key=lambda candidate: candidate['target_candidate_irr_diff_abs']
        )
        best_candidate = candidates[0]

        # Evaluate payment schedule results
        payment_schedule = {
            'monthly_payments': [],
            'ppmt_values': [],
            'ipmt_values': [],
            'pmt_values': [],
            'leftover_values': [],
        }

        current_leftover_value = car_price_uah + car_price_uah * commission_pct

        for month_idx in range(lease_term_months):
            ## Monthly payment
            payment_schedule['monthly_payments'].append(
                best_candidate['monthly_payments'][month_idx]
            )
            ## PPMT
            ppmt_value = npf.ppmt(
                best_candidate['rate'] / 12.0,
                month_idx + 1,
                lease_term_months,
                (car_price_uah + car_price_uah * commission_pct) * -1.0,
                fv=0
            )
            ppmt_value = float(ppmt_value)
            payment_schedule['ppmt_values'].append(ppmt_value)
            ## IPMT
            ipmt_value = npf.ipmt(
                best_candidate['rate'] / 12.0,
                month_idx + 1,
                lease_term_months,
                (car_price_uah + car_price_uah * commission_pct) * -1.0,
                fv=0
            )
            ipmt_value = float(ipmt_value)
            payment_schedule['ipmt_values'].append(ipmt_value)
            ## PMT
            pmt_value = npf.pmt(
                best_candidate['rate'] / 12.0,
                lease_term_months,
                (car_price_uah + car_price_uah * commission_pct) * -1.0,
                fv=0
            )
            pmt_value = float(pmt_value)
            payment_schedule['pmt_values'].append(pmt_value)
            ## Leftover value
            current_leftover_value -= ppmt_value
            payment_schedule['leftover_values'].append(current_leftover_value)

        # Calculate single value evaluations
        lease_sum_uah = sum(payment_schedule['pmt_values'])
        overpayment_uah = sum(payment_schedule['ipmt_values']) + (car_price_uah * commission_pct)
        single_value_evaluations = {
            'lease_sum_uah': lease_sum_uah,
            'overpayment_uah': overpayment_uah,
        }

        # Create final data storage
        pick_result = {
            'best_candidate': best_candidate,
            'payment_schedule': payment_schedule,
            'single_value_evaluations': single_value_evaluations,
        }

        return pick_result
