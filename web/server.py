from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class TicTacToeHandler(BaseHTTPRequestHandler):
    def address_string(self):
        return self.client_address[0]  # skip reverse DNS

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("web/index.html", "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self):
        if self.path == "/click":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            # log on server side
            print(f"Click received: row={data['row']} col={data['col']} figure={data['figure']}")
            # respond
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = { "status": "ok", "row": data['row'], "col": data['col'], "figure": data['figure'] }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), TicTacToeHandler)
    print("Server running on port 8080...")
    server.serve_forever()

