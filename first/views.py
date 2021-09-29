from django.shortcuts import render
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from django.core.files.storage import FileSystemStorage
import cv2

model = keras.models.load_model('E:\\Saumyaa\\facial_sentiment_analysis\\sentiment\\facefeatures_new_model.h5')

# Create your views here.
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img_height=224
    img_width=224
    img=image.load_img(testimage,target_size=(img_height,img_width))
    x= image.img_to_array(img)
    x=x/225
    x=x.reshape(1,img_height,img_width,3)
    pred = model.predict(x)
    import numpy as np
    l = ['ANGRY','FEAR','HAPPY','SAD','SURPRISE']
    x=np.argmax(pred)

    context = {'filePathName': filePathName,'pred':l[x]}
    return render(request, 'index.html', context)

