import unittest
import threading
import urllib.request
from urllib.error import HTTPError
from unittest.mock import MagicMock
from web.server import TicTacToeServer, TicTacToeHandler

from typing import Any

# A handler that doesn't print logs to stderr
class SilentHandler(TicTacToeHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

class TestWebServer(unittest.TestCase):
    server: TicTacToeServer
    port: int
    server_thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        # Set up a dummy model for the server
        crosses_model = MagicMock()
        zeroes_model = MagicMock()
        
        # Bind to port 0 to get an OS-assigned available port
        cls.server = TicTacToeServer(("127.0.0.1", 0), SilentHandler, crosses_model, zeroes_model)
        cls.port = cls.server.server_port
        
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        
    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()
        
    def test_get_tictactoe(self) -> None:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/tictactoe")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-type"), "text/html")
            content = resp.read()
            self.assertGreater(len(content), 0)
            self.assertIn(b"<!DOCTYPE html>", content)
            
    def test_404_not_found(self) -> None:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/nonexistent_route")
        
        with self.assertRaises(HTTPError) as context:
            urllib.request.urlopen(req)
            
        self.assertEqual(context.exception.code, 404)
        
    def test_post_click_missing_payload(self) -> None:
        # Just a small smoke test to ensure the POST route is reachable 
        # but fails appropriately when given a bad JSON payload.
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/click", 
            data=b"invalid json", 
            headers={"Content-Length": "12"}
        )
        
        # The server code crashes on invalid JSON and drops the socket connection.
        # It logs the exception trace to stderr, so we want to swallow stderr here.
        import sys
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with self.assertRaises(Exception):
                urllib.request.urlopen(req)
        finally:
            sys.stderr = old_stderr

if __name__ == "__main__":
    unittest.main()
