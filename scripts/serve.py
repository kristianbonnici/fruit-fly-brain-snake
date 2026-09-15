"""Serve the viewer locally and fetch verified replay chunks on demand."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit

from replay_data import ROOT, ReplayCache


def handler_for(cache):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route.startswith('/data/packed/'):
                name = route.removeprefix('/data/packed/')
                try:
                    path = cache.get(name)
                except FileNotFoundError as error:
                    self.send_error(404, str(error))
                    return
                except Exception:
                    self.send_error(502, 'Verified replay download failed. Check your connection and retry.')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(path.stat().st_size))
                self.end_headers()
                with path.open('rb') as source:
                    shutil.copyfileobj(source, self.wfile)
                return
            super().do_GET()

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def list_directory(self, path):
            self.send_error(404)

    return partial(Handler, directory=str(ROOT / 'viewer'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8802)
    parser.add_argument('--cache', type=Path, help='Optional existing verified chunk cache')
    parser.add_argument('--offline', action='store_true', help='Never download missing chunks')
    args = parser.parse_args()
    cache = ReplayCache(args.cache, args.offline)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(cache))
    print(f'Fruit fly Snake viewer: http://127.0.0.1:{server.server_port}/', flush=True)
    print('Offline mode.' if args.offline else 'Missing recordings download on demand.' if cache.origin
          else 'Using cached data only; set FRUIT_FLY_DATA_URL to enable downloads.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
