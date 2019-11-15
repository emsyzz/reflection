import requests


class SplashSender:

    def __init__(self) -> None:
        self.host = 'http://localhost:10000/'

    def send_immediate(self, command) -> None:
        try:
            r = requests.get(self.host + command)  # note that headers aren't 100% needed
            return r.json()  # j is now a python dict object
        except Exception as err:
            print("Cannot send to Splash: " + str(err))
