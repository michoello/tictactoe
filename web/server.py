from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random
from lib import tttc, tttp, pickup_model
from lib import game
from typing import cast, Any

import argparse
import math




class TicTacToeHandler(BaseHTTPRequestHandler):
    def address_string(self) -> str:
        return self.client_address[0]  # skip reverse DNS

    def do_GET(self) -> None:
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

    def do_POST(self) -> None:
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

            server = cast(TicTacToeServer, self.server)
            g = game.Game(server.model_x, server.model_o) ## TODO: , game_type, game_mode)
            b = game.Board()
            b.set(board)

            winner, winning_row = b.check_winner()

            response: dict[str, Any] = {"status": "ok"}
            if not winner:
                ply = -1 if human_plays == "X" else 1  ## who goes next

                # Step number is count of O's on the board. TODO: move it inside Game()
                turn_number = sum([1 for row in board for x in row if x == -1])
           
                prev_state = game.GameState(board=b, next_player=-ply, turn_number=turn_number - 1)
                next_state = g.choose_next_step(prev_state)
                x, y = next_state.prev_move if next_state.prev_move else (-1, -1)
                
                greedy_state = g.best_greedy_step(game.GameState(board=b, next_player=ply))
                
                if x is None or y is None:
                    print("sorry")

                response["row"] = x
                response["col"] = y
                assert greedy_state.policy is not None
                response["values"] = [
                    [round(v or -1, 2) for v in row] for row in greedy_state.policy
                ]

                print("Figure", human_plays)
                print("Model produced Values:")
                for row in response["values"]:
                    print(row)

                board[x][y] = ply 
                winner, winning_row = b.check_winner()

            # respond
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            if winner:
                response["winner"] = winner
                response["xyo"] = winning_row

            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")


class TicTacToeServer(HTTPServer):
    def __init__(self, server_address: tuple[str, int], RequestHandlerClass: Any, model_x: Any, model_o: Any) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.model_x = model_x
        self.model_o = model_o


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Server running a model")
    parser.add_argument("--model_x", type=str, help="Type and path of crosses model")
    parser.add_argument("--model_o", type=str, help="Type and path of zeroes model")
    args = parser.parse_args()

    crosses_master = pickup_model(*args.model_x.split(":"))
    model_x = crosses_master.model_x
    
    zeroes_master = pickup_model(*args.model_o.split(":"))
    model_o = zeroes_master.model_o
    
    server = TicTacToeServer(("0.0.0.0", 8080), TicTacToeHandler, model_x, model_o)
    print("Server running on port 8080...")
    server.serve_forever()
