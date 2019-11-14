import requests


class SplashSender:

    def __init__(self) -> None:
        self.host = 'http://localhost:10000/'

    def send_immediate(self, command) -> None:
        r = requests.get(self.host + command)  # note that headers aren't 100% needed
        return r.json()  # j is now a python dict object
