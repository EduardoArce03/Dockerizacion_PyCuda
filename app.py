#!/usr/bin/env python3

from flask import Flask
import pycuda.driver as drv
drv.init()

app = Flask(__name__)
@app.route("/")
def hello():
    res = "%d device(s) found." % (drv.Device.count())
    for ordinal in range(drv.Device.count()):
        dev = drv.Device(ordinal)
        res += ", Device #%d: %s" % (ordinal, dev.name())
        res += ", Compute Capability: %d.%d" % (dev.compute_capability())
        res += ", Total Memory: %s KB" % (dev.total_memory()//(1024))
    return res

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)

# Comandos para correr
# sudo docker build -t pycuda-app .
# sudo docker run --name mi-app-cuda3 --gpus all -p 5000:5000 pycuda-app