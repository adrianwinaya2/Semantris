from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from . import views

urlpatterns = [
    path("login", views.login_view, name="login"),
    path("logout", views.logout_view, name="logout"),
    path("register", views.register, name="register"),

    path("", views.index, name="index"),

    # path("arcade", views.arcade, name="arcade"),
    path("arcade/play", views.arcade_play, name="play_arcade"),
    path("arcade/highscore", views.arcade_highscore, name="arcade_highscore"),
]

urlpatterns += staticfiles_urlpatterns()