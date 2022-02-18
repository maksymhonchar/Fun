from django.shortcuts import render


def index(request):
    return render(request, 'index.html')


def car(request):
    return render(request, 'car.html')


def taxi(request):
    return render(request, 'taxi.html')


def cash(request):
    return render(request, 'cash.html')


def ir_all(request):
    return render(request, 'ir_all.html')


def ir_new(request):
    return render(request, 'ir_new.html')


def ir_one(request, ir_calculation_id: int):
    context = {
        'ir_calculation_id': ir_calculation_id
    }
    return render(request, 'ir_one.html', context=context)
