#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zzmind-pre-trainning 容器常驻进程：监控面板 + 进度文件

- HTTP :7791
    /            状态页（15s 自动刷新）
    /api/status  JSON 状态
    /log         最新训练日志 tail 64KB
- 每 30s 解析最新 /data/out/pretrain_*.log -> 写 /opt/zzmind/progress.json
  （宿主机 /home/ai-servers/zzmind/progress.json，start.sh 靠它决定从哪续训）
"""
import json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_DIR = '/data/out'
PROG_PATH = '/opt/zzmind/progress.json'
PORT = 7791
TOTAL = 183105   # 1.5e9 / 8192，与 start_pretrain 一致
LINE_RE = re.compile(r'step:(\d+)/(\d+) loss:([\d.]+) lr:[\d.eE+-]+ ([\d.]+)k tok/s ETA:(\d+)min')


def latest_log():
    try:
        fs = [f for f in os.listdir(LOG_DIR)
              if f.startswith('pretrain_') and f.endswith('.log')]
        return os.path.join(LOG_DIR, max(fs)) if fs else None
    except OSError:
        return None


def training_running():
    """扫 /proc 找 train_pretrain 进程（不依赖 pgrep）"""
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                if b'train_pretrain.py' in f.read():
                    return True
        except OSError:
            pass
    return False


def read_state():
    st = {'status': 'idle', 'step': 0, 'total': TOTAL, 'loss': None,
          'tok_s': None, 'eta_min': None, 'log': None,
          'updated': time.strftime('%F %T')}
    lg = latest_log()
    if lg:
        st['log'] = lg
        last = None
        try:
            with open(lg, encoding='utf-8', errors='replace') as f:
                for line in f:
                    if 'step:' in line:
                        last = line.strip()
        except OSError:
            pass
        if last:
            m = LINE_RE.search(last)
            if m:
                st['step'], st['total'] = int(m.group(1)), int(m.group(2))
                st['loss'] = float(m.group(3))
                st['tok_s'], st['eta_min'] = float(m.group(4)), int(m.group(5))
    if training_running():
        st['status'] = 'running'
    elif st['step'] >= st['total']:
        st['status'] = 'finished'
    return st


def write_progress():
    st = read_state()
    tmp = PROG_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, PROG_PATH)


def progress_loop():
    while True:
        try:
            write_progress()
        except Exception as e:
            print('progress write fail:', e, flush=True)
        time.sleep(30)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype='text/html; charset=utf-8', code=200):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith('/api/status'):
            self._send(json.dumps(read_state(), ensure_ascii=False),
                       'application/json; charset=utf-8')
        elif self.path.startswith('/log'):
            lg = latest_log()
            if not lg:
                self._send('no log yet', 'text/plain; charset=utf-8', 404)
                return
            with open(lg, 'rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 65536))
                self._send(f.read().decode('utf-8', 'replace'),
                           'text/plain; charset=utf-8')
        else:
            st = read_state()
            html = ('<html><head><meta charset="utf-8"><title>zzmind pretrain</title>'
                    '<meta http-equiv="refresh" content="15"></head>'
                    '<body style="font-family:monospace;background:#111;color:#0f0;padding:24px">'
                    f'<h2>zzmind-0.5B pretrain</h2>'
                    f'<p>status: <b>{st["status"]}</b>&nbsp; step: {st["step"]}/{st["total"]}'
                    f'&nbsp; loss: {st["loss"]}&nbsp; {st["tok_s"]}k tok/s&nbsp; ETA: {st["eta_min"]}min</p>'
                    f'<p><a href="/log" style="color:#8cf">/log tail</a> '
                    f'<a href="/api/status" style="color:#8cf">/api/status</a></p>'
                    '</body></html>')
            self._send(html)


if __name__ == '__main__':
    threading.Thread(target=progress_loop, daemon=True).start()
    write_progress()
    print(f'monitor listening on :{PORT}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
