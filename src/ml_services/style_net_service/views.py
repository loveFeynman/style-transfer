import cv2
from django.forms import Form
from django.http import HttpResponse
from django.http.multipartparser import MultiPartParser
from django.shortcuts import render, redirect
import sys,os
import numpy as np

from django.template.loader import get_template, render_to_string
from django.views.decorators.csrf import csrf_exempt



sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import constants
import full_graph_stylizer_network
import graph_stylizer_network
import model_utilities
from image_utilities import *


service = full_graph_stylizer_network.StyleNetService(model_utilities.get_most_recent_model_name(constants.MODELS_DIR, graph_stylizer_network.model_prefix))
service.start()


@csrf_exempt
def index(request):
    if request.method == 'POST':

        file_name = service.model_name + '_' + request.FILES['file'].name
        data = request.FILES['file'].read()

        img = np.fromstring(data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = flip_BR(img)
        img = pixel_to_decimal(img)
        img = image_3d_to_4d(img)

        omg = service.run_on_image(img)

        save_image(omg, file_name)

        return HttpResponse(file_name)
    else:
        text = render_to_string('upload.html')
        return HttpResponse(text)

@csrf_exempt
def get_file(request):
    if request.method == 'GET':
        file_name = request.GET.get('file','')

        with open(file_name, 'rb') as fl:
            odata = fl.read()

        os.remove(file_name)

        response = HttpResponse(odata, content_type='application/force-download')  # content_type='image/jpeg')
        response['Content-Disposition'] = 'attachment; filename="' + file_name + '"'

        return response
    else:
        return HttpResponse('Nothing here.')