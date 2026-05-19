# HuggingFace Spaces entrypoint
# This file is required by HF Spaces Docker SDK
import subprocess
import os

# Start the Node.js backend
subprocess.run(["node", "src/server.js"])
