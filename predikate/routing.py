from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from website.consumers import NoseyConsumer

application = ProtocolTypeRouter({
    "websocket": URLRouter([
        path("notifications/", NoseyConsumer),
    ])
})