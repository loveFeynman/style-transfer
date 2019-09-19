import cv2
from django.forms import Form
from django.http import HttpResponse
from django.http.multipartparser import MultiPartParser
from django.shortcuts import render
import sys,os
import numpy as np

from django.template.loader import get_template, render_to_string
from django.views.decorators.csrf import csrf_exempt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import constants
import graph_stylizer_network
import model_utilities
from image_utilities import *


service = graph_stylizer_network.StyleNetService(model_utilities.get_most_recent_model_name(constants.MODELS_DIR, graph_stylizer_network.model_prefix))
service.start()

@csrf_exempt
def index(request):
    if request.method == 'POST':
        data = request.FILES['file'].read()

        img = np.fromstring(data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = flip_BR(img)
        img = pixel_to_decimal(img)
        img = image_3d_to_4d(img)

        omg = service.run_on_image(img)


        imfile = 'temp_transformed_image.jpg'
        save_image(omg, imfile)

        return HttpResponse('uploaded')
    else:
        text = render_to_string('upload.html')
        return HttpResponse(text)

@csrf_exempt
def get_file(request):
    imfile = 'temp_transformed_image.jpg'

    with open(imfile, 'rb') as fl:
        odata = fl.read()

    print('done encoding')

    response = HttpResponse(odata, content_type='application/force-download')  # content_type='image/jpeg')
    response['Content-Disposition'] = 'attachment; filename="' + imfile + '"'
    # response = HttpResponse('helloooooo')

    print('sending image')

    return response