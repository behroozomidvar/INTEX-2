import uvicorn
from server.main import *

uvicorn.run(app, host="0.0.0.0", port=8080)
