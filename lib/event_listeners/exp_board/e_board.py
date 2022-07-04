import os
import time
from threading import Thread
import re

from socketserver import ThreadingMixIn
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from base64 import urlsafe_b64decode, urlsafe_b64encode

from experimenter import e
import json


def replace_all(x, config):
    # not very pretty code, but it works
    # replaces all ${key} -> config[key] in the attr.

    wrap = {'txt': x, 'offset': 0}
    regex = "(\$\{(\w(\w|\\n)*)\})"

    def replace(match, o):
        txt = o['txt']
        offset = o['offset']
        k = match.group(2)
        s = match.span(2)
        if k not in config:
            raise Exception('Config "' + k + '" not found !')

        o['txt'] = txt[: offset + s[0] - 2] + config[k] + txt[offset + s[1] + 1:]
        # offset is used to compensate the "original" spans
        # for the differences in the string before and after
        o['offset'] += len(config[k]) - (s[1] - s[0]) - 3

    ''
    [replace(x, wrap) for x in re.finditer(regex, x)]
    return wrap['txt']


def HandlerFactory(get_data):
    class CustomHandler(SimpleHTTPRequestHandler):
        """
            created per request.
            wrapper HandlerFactory is used to pass the data object.
        """

        def __init__(self, *args, **kwargs):
            self.routes = {
                '/index.html': self.index,
                '/api': self.api
            }
            self.data = get_data()
            super().__init__(*args, directory=e.ws() + 'outputs/', **kwargs)

        def html(self, path, data=None):
            full_path = os.path.dirname(__file__) + '/' + path
            f = open(full_path, 'r')

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(f.read(), "utf-8"))

        def json(self, data=None):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(data), 'utf-8'))

        def api(self, query):
            exp_key = urlsafe_b64decode(query['k'][0]).decode() \
                if 'k' in query else e.key()
            data = {
                **self.data,
                'key': exp_key,
                'files': [exp_key + '/out/' + f for f in os.listdir(e.ws() + 'outputs/' + exp_key + '/out/')],
                'experiments': [urlsafe_b64encode(bytes(f, 'utf-8')).decode() for f in os.listdir(e.ws() + 'outputs/')]
            }
            self.json(data)

        def index(self, query):
            self.html('/index.html')

        def do_GET(self):
            parsed_url = urlparse(self.path)
            query = parse_qs(parsed_url.query)
            path = parsed_url.path
            path = '/index.html' if path == '/' else path

            if path in self.routes:
                self.routes[path](query)
            else:
                super().do_GET()

        def log_message(self, format, *args):
            return

    return CustomHandler


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class EBoard:

    def __init__(self, port=8080):
        hostName = "0.0.0.0"
        serverPort = port
        self.server = None
        self.data = {
            'tick': 0
        }
        self.thread = Thread(target=self.serve_on_port, args=[
            hostName,
            serverPort
        ])
        self.thread.start()

    def serve_on_port(self, hostname, port):
        handler = HandlerFactory(self._get_data)
        self.server = ThreadingHTTPServer((hostname, port), handler)
        print("Server started http://%s:%s" % (hostname, port))
        self.server.serve_forever()

    def _get_data(self):
        return self.data

    def on__all(self, ev):
        self.data = {
            'tick': self.data['tick'] + 1
        }

    def on_e_end(self, ev):
        self.server.server_close()
        self.thread.join()
