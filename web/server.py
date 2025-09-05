from http.server import BaseHTTPRequestHandler, HTTPServer
import os

class TicTacToeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("web/index.html", "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

class HelloHandler(BaseHTTPRequestHandler):
 def do_GET(self):
        self.send_response(200)  # HTTP status 200 OK
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"yopta!\n")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), TicTacToeHandler)
    #server = HTTPServer(("0.0.0.0", 8080), HelloHandler)
    print("Server starting on port 8080...")
    server.serve_forever()
