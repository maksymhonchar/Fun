import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .ratepicker import RatePicker


@require_http_methods(['POST', ])
@csrf_exempt
def view_pick_v2(request):
    # Fetch POST fields from request
    request_post_data = json.loads(
        request.body.decode('utf-8')
    )

    # Prepare data storages for rate picker
    customer_metrics = {
        'car_price_usd': request_post_data.get('car_price_usd', None),
        'down_payment_usd': request_post_data.get('down_payment_usd', None),
        'lease_term_months': request_post_data.get('lease_term_months', None),
        'discount_pct': request_post_data.get('discount_pct', 0),
    }
    business_metrics = {
        'commission_pct': request_post_data.get('commission_pct', None),
        'tracker_price_uah': request_post_data.get('tracker_price_uah', None),
        'tracker_subscription_fee_uah': request_post_data.get('tracker_subscription_fee_uah', None),
        'insurance_pct': request_post_data.get('insurance_pct', None),
        'desired_irr_pct': request_post_data.get('desired_irr_pct', None),
    }
    misc_metrics = {
        'exchange_rate': request_post_data.get('exchange_rate', None),
        'precision': request_post_data.get('precision', None)
    }

    # Pick the best lease rate
    rate_picker = RatePicker()
    rate_picker_result = rate_picker.pick_v2(
        customer_metrics=customer_metrics,
        business_metrics=business_metrics,
        misc_metrics=misc_metrics
    )

    # Define view response
    response_data = {
        'status': 'ok',
        'result': rate_picker_result
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response


@require_http_methods(['POST', ])
@csrf_exempt
def view_pick_cash_credit(request):
    # Fetch POST fields from request
    request_post_data = json.loads(
        request.body.decode('utf-8')
    )

    # Prepare data storages for rate picker
    customer_metrics = {
        'car_price_uah': request_post_data.get('car_price_uah', None),
        'lease_term_months': request_post_data.get('lease_term_months', None),
    }
    business_metrics = {
        'commission_pct': request_post_data.get('commission_pct', None),
        'desired_irr_pct': request_post_data.get('desired_irr_pct', None),
    }
    misc_metrics = {
        'precision': request_post_data.get('precision', None)
    }

    # Pick the best lease rate
    rate_picker = RatePicker()
    rate_picker_result = rate_picker.pick_cash_credit(
        customer_metrics=customer_metrics,
        business_metrics=business_metrics,
        misc_metrics=misc_metrics
    )

    # Define view response
    response_data = {
        'status': 'ok',
        'result': rate_picker_result
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response
