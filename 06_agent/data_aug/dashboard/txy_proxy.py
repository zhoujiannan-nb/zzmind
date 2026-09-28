#!/usr/bin/env python3
"""txy 看板服务：静态目录 + /api/* 反向代理到本机 7799（ssh 隧道 → node05 生成器）"""
import http.server, socketserver, urllib.request, os, sys

PORT = 7788
STATIC = os.path.dirname(os.path.abspath(__file__))
UP = "http://127.0.0.1:7799"

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=STATIC, **kw)

    def do_POST(self):
        self._proxy("POST")

    def do_OPTIONS(self):
        self._proxy("OPTIONS")

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy("GET")
        else:
            super().do_GET()

    def _proxy(self, method):
        ln = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(ln) if ln else None
        url = UP + self.path
        req = urllib.request.Request(url, data=body, method=method)
        for k in ("Content-Type", "Authorization"):
            if self.headers.get(k):
                req.add_header(k, self.headers[k])
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                data = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            err = f'{{"err":"{e}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def log_message(self, *a):
        pass

socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), H) as srv:
    print(f"dash proxy on :{PORT} -> {UP}", flush=True)
    srv.serve_forever()
