#!/usr/bin/env python3
"""Live training log viewer — serves last N lines with auto-refresh."""

import os
from http.server import HTTPServer, BaseHTTPRequestHandler

LOG_PATH = "/home/stingraider/ic-6dof-arm/logs/isaac/phase_0_live.log"
PORT = 8090

class LogHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            lines = []
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r") as f:
                    all_lines = f.readlines()
                tail = all_lines[-40:]
                lines = [l.rstrip("\n") for l in tail]
            else:
                lines = ["[Log file not found yet]"]
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phase 0 Training Log</title>
<meta http-equiv="refresh" content="5">
<style>
  body {{ background:#0d1117; color:#c9d1d9; font:14px/1.5 'JetBrains Mono','Fira Code',monospace; padding:16px; }}
  pre {{ margin:0; white-space:pre-wrap; }}
  .hl-reach {{ color:#58a6ff; }}
  .hl-iter {{ color:#f0883e; }}
  .hl-reward {{ color:#3fb950; }}
  .hl-time {{ color:#8b949e; }}
  .title {{ color:#f0c040; font-size:18px; font-weight:bold; margin-bottom:12px; }}
  .footer {{ color:#484f58; font-size:12px; margin-top:12px; }}
</style></head><body>
<div class="title">╰( ⁰ ਊ ⁰ )━☆ Phase 0 — Live Log</div>
<pre>"""
            for line in lines:
                html += self._colorize(line) + "\n"
            html += """</pre>
<div class="footer">Auto-refresh every 5s · Tailscale: http://100.xxx:8090</div>
</body></html>"""
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _colorize(self, line):
        import re
        line = re.sub(r"(iteration \d+/\d+)", r'<span class="hl-iter">\1</span>', line, flags=re.IGNORECASE)
        line = re.sub(r"(Episode_Reward/reach: [\d.]+)", r'<span class="hl-reach">\1</span>', line)
        line = re.sub(r"(Mean reward: [\d.]+)", r'<span class="hl-reward">\1</span>', line)
        line = re.sub(r"(Time elapsed: [\d:]+)", r'<span class="hl-time">\1</span>', line)
        line = re.sub(r"(ETA: [\d:]+)", r'<span class="hl-time">\1</span>', line)
        return line

    def log_message(self, format, *args):
        pass  # Quiet

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), LogHandler)
    print(f"📡 Live log server at http://0.0.0.0:{PORT}/")
    print(f"   Watching: {LOG_PATH}")
    print(f"   Access via Tailscale: http://<tailscale-ip>:{PORT}/")
    server.serve_forever()
