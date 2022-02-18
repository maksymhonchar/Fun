import ast
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import IRCalculation
from .irpicker import IRDetailsHandler


@require_http_methods(["GET", ])
def view_ir_calculations(request):
    # Fetch all IRCalculation objects
    ir_calculations_queryset = IRCalculation.objects.order_by('-create_time').all()
    ir_calculations_payloads = [
        [
            calculation.id,
            calculation.create_time,
            ast.literal_eval(calculation.payload)
        ]
        for calculation in ir_calculations_queryset
    ]

    # Define view response
    response_data = {
        'status': 'ok',
        'result': ir_calculations_payloads
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response


@require_http_methods(["GET", ])
def view_ir_calculation(request):
    # Fetch GET parameters from request
    ir_calculation_id = request.GET.get("ir_calculation_id", -1)

    # Validate parameters
    if ir_calculation_id == -1:
        # Define view response
        response_data = {
            'status': 'failed',
            'result': '[ir_calculation_id] is missing or -1'
        }
        view_response = JsonResponse(response_data)

        # Return response
        return view_response

    # Request specific IR calculation
    ir_calculation_queryset = IRCalculation.objects.filter(id=ir_calculation_id)

    # Check whether specific IR calculation exists
    if ir_calculation_queryset.count() == 0:
        # Define view response
        response_data = {
            'status': 'failed',
            'result': f'[ir_calculation_id]={ir_calculation_id} does not exist'
        }
        view_response = JsonResponse(response_data)

        # Return response
        return view_response

    # Fetch payload for specific IR calculation
    ir_calculation_payload = [
        ir_calculation_queryset.first().create_time,
        ast.literal_eval(
            ir_calculation_queryset.first().payload
        )
    ]

    # Define view response
    response_data = {
        'status': 'ok',
        'result': ir_calculation_payload
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response


@require_http_methods(["POST", ])
@csrf_exempt
def view_new_calculation(request):
    # Fetch POST fields from request
    request_post_data = json.loads(
        request.body.decode('utf-8')
    )

    # Prepare data storages
    loan_details = request_post_data.get('loan_details', {})
    income_details = request_post_data.get('income_details', {})
    expenses_details = request_post_data.get('expenses_details', {})

    # Evaluate IR details
    ir_details_handler = IRDetailsHandler(
        loan_details=loan_details,
        income_details=income_details,
        expenses_details=expenses_details
    )
    ir_details = ir_details_handler.handle()

    # Save IR evaluation details
    new_calcuation_data = {}
    new_calcuation_data['inputs'] = request_post_data
    new_calcuation_data['evaluations'] = ir_details
    new_calculation_db_object = IRCalculation(payload=new_calcuation_data)
    new_calculation_db_object.save()
    new_calcuation_data['new_db_object_id'] = new_calculation_db_object.id

    # Define view response
    response_data = {
        'status': 'ok',
        'result': new_calcuation_data
    }
    view_response = JsonResponse(response_data)

    # Return response
    return view_response
