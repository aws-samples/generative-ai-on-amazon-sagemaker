import json
import boto3
import subprocess
import numpy as np
from pip._internal import main
import sys
import os
from PIL import Image, ImageChops, ImageEnhance


def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = '/tmp/tempresaved.jpg'
    im = Image.open(filename)
    bm = im.convert('RGB')
    im.close()
    im=bm
    im.save(resaved_filename, 'JPEG', quality = quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    im.close()
    bm.close()
    resaved_im.close()
    del filename
    del resaved_filename
    del im
    del bm
    del resaved_im
    del extrema
    del max_diff
    del scale
    return ela_im


def check_image(image, endpoint_name):
    X = []
    X.append(np.array(convert_to_ela_image(image, 90).resize((128, 128))).flatten() / 255.0)
    X = np.array(X)

    X = X.reshape(-1, 128, 128, 3)
    
    data = {'instances': X.tolist()}
    sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
    # Gets inference from the model hosted at the specified endpoint:
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, 
        Body=json.dumps(data),
        ContentType="application/json"
        )

    # Decodes and prints the response body:
    #print(response['Body'].read().decode('utf-8'))
    predictions_res = json.loads(response['Body'].read().decode('utf-8'))
    predictions = predictions_res['predictions']
    
    pred_classes = np.argmax(predictions,axis = 1) 
    return pred_classes.tolist()[0]