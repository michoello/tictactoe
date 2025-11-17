from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random
from lib import tttc, tttp, pickup_model
from lib import game

import argparse
import math

parser = argparse.ArgumentParser(description="Web Server running a model")
parser.add_argument("--crosses_model", type=str, help="Type and path of crosses model")
parser.add_argument("--zeroes_model", type=str, help="Type and path of zeroes model")
args = parser.parse_args()


class TicTacToeHandler(BaseHTTPRequestHandler):
    def address_string(self):
        return self.client_address[0]  # skip reverse DNS

    def do_GET(self):
        if self.path.startswith("/tictactoe"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("web/index.html", "rb") as f:
                self.wfile.write(f.read())
        elif self.path.startswith("/ico"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            if self.path.endswith(".ico"):
                path = "web" + self.path
            else:
                path = "web/ico/index.html"
            print(self.path, " ", path)
            with open(path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "gtfo por favor")

    def do_POST(self):
        if self.path == "/click":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            human_plays = data['human_plays']  # what human plays

            # log on server side
            print(f"Click: row={data['row']} col={data['col']} human_plays={data['human_plays']}")
            print("Board state:")
            board = data["board"]
            for row in board:
                print(row)

            # Choosing the best next step with model!
            winner, xyo = game.Board(board).check_winner()
            response = {"status": "ok"}
            if not winner:
                ply = -1 if human_plays == "X" else 1  ## who goes next
                boards = game.Board(board).all_next_steps(ply)
                if len(boards) == 0:
                    print("sorry")

                # Step number is count of zeroes on the board
                step_no = sum([1 for row in board for x in row if x == -1])

                boards = [(b[0].board, b[1], b[2]) for b in boards]
                if human_plays == "O":
                  values = self.server.m_crosses.get_next_step_values(boards)
                else:
                  values = self.server.m_zeroes.get_next_step_values(boards)
                exploration_rate = 0.0
                x, y = game.choose_next_step(values, ply, step_no, exploration_rate)

                response["row"] = x
                response["col"] = y
                response["values"] = [
                    [round(v or -1, 2) for v in row] for row in values
                ]

                print("Figure", human_plays)
                print("Model produced Values:")
                for row in response["values"]:
                    print(row)

                if human_plays == "X":
                  board[x][y] = -1  # zero
                else:
                  board[x][y] = 1  # cross
                winner, xyo = game.Board(board).check_winner()

            # respond
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            if winner:
                response["winner"] = winner
                response["xyo"] = xyo

            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")


class TicTacToeServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, crosses_model, zeroes_model):
        super().__init__(server_address, RequestHandlerClass)
        self.m_crosses = crosses_model
        self.m_zeroes = zeroes_model


if __name__ == "__main__":
    crosses_model = pickup_model(*args.crosses_model.split(":"))
    zeroes_model = pickup_model(*args.zeroes_model.split(":"))
    server = TicTacToeServer(("0.0.0.0", 8080), TicTacToeHandler, crosses_model, zeroes_model)
    print("Server running on port 8080...")
    server.serve_forever()
