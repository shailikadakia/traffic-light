from flask import Flask, request
from flask_cors import CORS  # <--- ADD THIS
import json

app = Flask(__name__)
CORS(app)  # <--- ENABLE CORS

@app.route("/save_coordinates", methods=["POST"])
def save_coordinates():
    data = request.get_json()
    print("📍 Coordinates received:", data)
    with open("selected_coordinates.json", "w") as f:
        json.dump(data, f)
    return {"status": "saved"}

if __name__ == "__main__":
    app.run(port=5000)
