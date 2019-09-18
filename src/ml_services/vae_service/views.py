from django.http import HttpResponse
from django.shortcuts import render
import sys,os

from django.views.decorators.csrf import csrf_exempt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), './constants.py'))
# Create your views here.
# from vae import VAEService
#
# vae_service = VAEService('VAE_2019-07-12-14-51')
# vae_service.start()

image_queue = []

@csrf_exempt
def index(request):
    if request.method == 'POST':
        print(request.body)
    else:
        print(request.GET)
    response = HttpResponse("hello from views")
    response['Access-Control-Allow-Origin'] = 'http://localhost:8001'
    return response