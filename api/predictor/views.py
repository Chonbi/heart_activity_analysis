from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse
from rest_framework.views import APIView
import json

class call_model(APIView):
    def get(self,request):
        if request.method == 'GET':
            # get sound from request
            sound = request.GET.get('sound')
   
            # vectorize sound
            vector = PredictorConfig.vectorizer.transform([sound])
            # predict based on vector
            prediction = PredictorConfig.regressor.predict(vector)[0]
            # build response
            response = {'dog': prediction}
            # return response
            return JsonResponse(response)

class logisticregression(APIView):
    def get(self, request):
        #PredictorConfig.logistic_regression.predict()
        return JsonResponse({'prediction': accuracy})

class xgboost(APIView):
    def get(self, request):
        print 'Raw Data: "%s"' % request.body
        obj = json.loads(request.body)
        print obj['coucou']
        return JsonResponse({'ok':'alice'})

class randomforest(APIView):
    def get(self, request):
        print 'Raw Data: "%s"' % request.body
        obj = json.loads(request.body)
        print obj['coucou']
        return JsonResponse({'ok':'alice'})