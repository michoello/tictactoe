from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random
from lib import tttc, tttp, pickup_model
from lib import game

import argparse

parser = argparse.ArgumentParser(description="Web Server running a model")
parser.add_argument("--zeroes_model", type=str, help="Type and path of zeroes model")
args = parser.parse_args()


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
            print(f"Click: row={data['row']} col={data['col']} figure={data['figure']}")
            print("Board state:")
            board = data["board"]
            for row in board:
                print(row)

            # Choosing the best next step with model!
            ply = -1 # zeroes go next
            boards = game.Board(board).all_next_steps(ply)
            if len(boards) == 0:
                print("sorry")

            boards = [(b[0].board, b[1], b[2]) for b in boards]
            values = self.server.m_zeroes.get_next_step_values(boards)
            exploration_rate = 0.0
            step_no = 100 # to eliminate randomness
            ply = game.choose_next_step(values, -1, step_no, exploration_rate)

            # respond
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = { "status": "ok", "row": ply[0], "col": ply[1] } # "received": data }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")


class TicTacToeServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, zeroes_model):
        super().__init__(server_address, RequestHandlerClass)
        self.m_zeroes = zeroes_model



if __name__ == "__main__":
    print("YOOOOO", args.zeroes_model.split(":"))
    zeroes_model = pickup_model(*args.zeroes_model.split(":"))
    server = TicTacToeServer(("0.0.0.0", 8080), TicTacToeHandler, zeroes_model)
    print("Server running on port 8080...")
    server.serve_forever()

