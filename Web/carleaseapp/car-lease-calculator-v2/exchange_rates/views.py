import ast

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from .models import CurrencyRateRequest
from .nbuhandler import NBUHandler


@require_http_methods(["GET"])
def view_nbu(request):
    # Decide whether to update currency rate data
    cache_minutes = 60
    min_create_time = timezone.now() - timezone.timedelta(minutes=cache_minutes)
    fresh_nbu_exchange_rates = CurrencyRateRequest.objects.filter(create_time__gte=min_create_time)
    
    # Fetch fresh data if it exists
    if fresh_nbu_exchange_rates.count() == 0:
        nbu_handler = NBUHandler()
        nbu_exchange_rates_payload = nbu_handler.get_exchange_rates()
        currency_rate_request_object = CurrencyRateRequest(
            source="National Bank of Ukraine",
            payload=str(nbu_exchange_rates_payload)
        )
        currency_rate_request_object.save()
    else:
        last_nbu_exchange_rates_queryset = fresh_nbu_exchange_rates.order_by('-create_time').first()
        nbu_exchange_rates_payload = ast.literal_eval(last_nbu_exchange_rates_queryset.payload)

    # Find out USD-UAH exchange rate
    usd_uah_exchange_rate = None
    for exchange_rate_data in nbu_exchange_rates_payload:
        if exchange_rate_data["cc"] == "USD":
            usd_uah_exchange_rate = exchange_rate_data
            break

    # Define view response
    response_data = {
        'status': 'ok',
        'rate': usd_uah_exchange_rate
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response
