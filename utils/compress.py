import zlib
import base64

def compress(s: str) -> str:
  return base64.b64encode(zlib.compress(s.encode('utf-8'))).decode('ascii')

def decompress(s: str) -> str:
  return zlib.decompress(base64.b64decode(s.encode('ascii'))).decode('utf-8')
