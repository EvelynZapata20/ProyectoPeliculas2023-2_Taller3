from django.shortcuts import render

# Create your views here.
from movie.models import Movie

from dotenv import load_dotenv, find_dotenv
import json
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np

def recommendations(request):
    req = request.GET.get('searchMovie')

    if req:
    # Carga los datos de las películas
        _ = load_dotenv('../openAI.env')
        openai.api_key = os.environ['openAI_api_key']

        with open('../movie_descriptions_embeddings.json', 'r') as file:
            file_content = file.read()
            movies = json.loads(file_content)
        
        emb = get_embedding(req,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(movies)):
            sim.append(cosine_similarity(emb,movies[i]['embedding']))
        sim = np.array(sim)
        idx = np.argmax(sim)
        
        movs= Movie.objects.filter(title__icontains=movies[idx]['title'])

    else:
        movs = Movie.objects.all()
    
    return render(request, 'recommendations.html', {'searchMovie':req, 'movies': movs})