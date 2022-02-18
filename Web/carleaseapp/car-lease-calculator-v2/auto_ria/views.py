import math
import requests

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods


@require_http_methods(["GET", ])
def view_marks(request):
    # Fetch GET parameters from request
    autoria_category_id = request.GET.get("category_id", -1)

    # Validate parameters
    if autoria_category_id == -1:
        # Define view response
        response_data = {
            'status': 'failed',
            'result': '[category_id] is missing or -1'
        }
        view_response = JsonResponse(response_data)

        # Return response
        return view_response

    # Fetch marks from AUTO.RIA
    autoria_api_key = settings.AUTORIA_API_KEY
    autoria_marks_url = f"https://developers.ria.com/auto/categories/{autoria_category_id}/marks?api_key={autoria_api_key}"
    autoria_response = requests.get(url=autoria_marks_url)
    autoria_response_json = autoria_response.json()

    # Define view response
    response_data = {
        'status': 'ok',
        'result': autoria_response_json
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response


@require_http_methods(["GET", ])
def view_models(request):
    # Fetch GET parameters from request
    autoria_category_id = request.GET.get("category_id", -1)
    autoria_mark_id = request.GET.get("mark_id", -1)

    # Validate parameters
    if (autoria_category_id == -1) or (autoria_mark_id == -1):
        # Define view response
        response_data = {
            'status': 'failed',
            'result': 'either [category_id] or [mark_id] is missing or -1'
        }
        view_response = JsonResponse(response_data)

        # Return response
        return view_response

    # Fetch marks from AUTO.RIA
    autoria_api_key = settings.AUTORIA_API_KEY
    autoria_models_url = f"http://api.auto.ria.com/categories/{autoria_category_id}/marks/{autoria_mark_id}/models?api_key={autoria_api_key}"
    autoria_response = requests.get(url=autoria_models_url)
    autoria_response_json = autoria_response.json()

    # Define view response
    response_data = {
        'status': 'ok',
        'result': autoria_response_json
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response


@require_http_methods(["GET", ])
def view_average_price(request):
    # Fetch GET parameters from request
    model_id = request.GET.get("model_id", "-1")
    transmission_type_id = request.GET.get("transmission_type_id", "-1")
    fuel_type_id = request.GET.get("fuel_type_id", "-1")
    production_year = int(request.GET.get("production_year", 2021))
    years_to_analyze = int(request.GET.get("years_to_analyze", "failed"))
    exchange_rate = float(request.GET.get("usd_uah_rate", "failed"))

    # Build AUTO.RIA request URL
    autoria_api_key = settings.AUTORIA_API_KEY
    autoria_average_price_url = f"https://developers.ria.com/auto/average_price?api_key={autoria_api_key}"
    if model_id != "-1":
        autoria_average_price_url += "&model_id=" + model_id
    if transmission_type_id != "-1":
        autoria_average_price_url += "&gear_id=" + transmission_type_id
    if fuel_type_id != "-1":
        autoria_average_price_url += "&fuel_id=" + fuel_type_id

    # Fetch N AUTO.RIA requests
    autoria_average_price_responses = []
    autoria_years = []
    autoria_avg_prices = []
    for current_production_year in range(years_to_analyze + 1):
        ##
        current_autoria_average_price_url = autoria_average_price_url + "&yers={0}".format(production_year - current_production_year)
        ##
        autoria_average_price_response = requests.get(url=current_autoria_average_price_url).json()
        autoria_average_price_responses.append(
            autoria_average_price_response
        )
        ##
        autoria_years.append(
            str(production_year - current_production_year)
        )
        ##
        median_value = float(autoria_average_price_response["percentiles"]["50.0"])
        if math.isnan(median_value):
            median_value = 0.0
        autoria_avg_prices.append(
            median_value * exchange_rate
        )

    # Create result storage
    autoria_average_price_response = {
        'autoria_average_price_responses': autoria_average_price_responses,
        'autoria_years': autoria_years,
        'autoria_avg_prices': autoria_avg_prices,
    }

    # Define view response
    response_data = {
        'status': 'ok',
        'result': autoria_average_price_response
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response
