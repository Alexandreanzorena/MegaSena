# Google Colab: envie a pasta megasena_app (ou o zip), depois execute numa célula:
#   !pip -q install flask openpyxl
#   %run /content/megasena_app/colab_run.py
import sys, threading
sys.path.insert(0, "/content/megasena_app")
import app
threading.Thread(target=lambda: app.app.run(host="0.0.0.0", port=8000), daemon=True).start()
from google.colab import output
output.serve_kernel_port_as_window(8000)
