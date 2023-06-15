from django.shortcuts import render
from .utils import get_sorted_dict

# Create your views here.

def login_view():
    pass

def logout_view():
    pass

def register():
    pass

def index(request):
    sorted_dict = get_sorted_dict()
    return render(request, "index.html", {'sorted_dict': sorted_dict})

def arcade_play():
    pass

def arcade_highscore():
    pass