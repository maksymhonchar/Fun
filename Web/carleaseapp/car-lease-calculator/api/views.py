import ast
import datetime
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .currencyrates import get_currency_to_uah_nbu_exchange_rate, get_nbu_exchange_rates
from .finpicking import find_rate_having_irr
from .models import CurrencyRateRequest


@require_http_methods(["GET", ])
def view_rates(request):
    force_update = request.GET.get('force_update', None)
    if force_update == 'True':
        nbu_exchange_rates = get_nbu_exchange_rates()
        currency_rate_request_object = CurrencyRateRequest(
            source="Національний банк України",
            payload=str(nbu_exchange_rates)
        )
        currency_rate_request_object.save()
    else:
        cache_minutes = 30
        min_create_time = datetime.datetime.now() - datetime.timedelta(minutes=cache_minutes)
        fresh_nbu_exchange_rates = CurrencyRateRequest.objects.filter(
            create_time__gte=min_create_time
        )
        if fresh_nbu_exchange_rates.count() == 0:
            nbu_exchange_rates = get_nbu_exchange_rates()
            currency_rate_request_object = CurrencyRateRequest(
                source="Національний банк України",
                payload=str(nbu_exchange_rates)
            )
            currency_rate_request_object.save()
        else:
            last_nbu_exchange_rate = fresh_nbu_exchange_rates.order_by('-create_time')[0]
            nbu_exchange_rates = ast.literal_eval(last_nbu_exchange_rate.payload)

    data = {
        'status': 'ok',
        'rates': nbu_exchange_rates,
    }
    params = {
        'ensure_ascii': False
    }
    return JsonResponse(data, json_dumps_params=params)


@require_http_methods(["GET", ])
@csrf_exempt
def view_rates_currency_code(request, currency_code):
    cache_minutes = 30
    min_create_time = datetime.datetime.now() - datetime.timedelta(minutes=cache_minutes)
    fresh_nbu_exchange_rates = CurrencyRateRequest.objects.filter(
        create_time__gte=min_create_time
    )
    if fresh_nbu_exchange_rates.count() == 0:
        nbu_exchange_rates = get_nbu_exchange_rates()
        currency_rate_request_object = CurrencyRateRequest(
            source="Національний банк України",
            payload=str(nbu_exchange_rates)
        )
        currency_rate_request_object.save()
    else:
        last_nbu_exchange_rate = fresh_nbu_exchange_rates.order_by('-create_time')[0]
        nbu_exchange_rates = ast.literal_eval(last_nbu_exchange_rate.payload)

    rate = get_currency_to_uah_nbu_exchange_rate(
        nbu_exchange_rates=nbu_exchange_rates,
        currency_code=currency_code
    )

    data = {
        'status': 'ok',
        'rate': rate,
    }
    params = {
        'ensure_ascii': False
    }
    return JsonResponse(data, json_dumps_params=params)


@require_http_methods(["POST", ])
@csrf_exempt
def view_pick_find_rate_having_irr(request):
    request_post_data = json.loads(
        request.body.decode('utf-8')
    )
    price = request_post_data.get('price', None)
    upfront_payment = request_post_data.get('upfront_payment', None)
    term = request_post_data.get('term', None)
    commission_rate = request_post_data.get('commission_rate', None)
    target_irr = request_post_data.get('target_irr', None)
    precision = request_post_data.get('precision', None)

    if precision is None:
        precision = 1
    if any([value == None for value in [price, upfront_payment, term, commission_rate, target_irr]]):
        data = {
            'status': 'failed',
            'description': 'either price, upfront_payment, term, commission_rate or target_irr is None or not found'
        }
        return JsonResponse(data)
    else:
        customer_metrics = {
            'price': price,
            'upfront_payment': upfront_payment,
            'term': term
        }
        business_metrics = {
            'commission_rate': commission_rate,
            'target_irr': target_irr
        }
        candidates, best_candidate = find_rate_having_irr(
            customer_metrics=customer_metrics,
            business_metrics=business_metrics,
            precision=precision
        )
        data = {
            'status': 'ok',
            'candidates': candidates,
            'best_candidate': best_candidate
        }
        return JsonResponse(data)
