from ryu.base import app_manager
import logging

LOG = logging.getLogger("dummy_app")

class DummyApp(app_manager.RyuApp):
    def __init__(self, *args, **kwargs):
        super(DummyApp, self).__init__(*args, **kwargs)
        LOG.info("Dummy app loaded successfully without Eventlet errors!")
