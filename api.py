from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the data from the JSON file
with open("data.json", "r") as file:
    data = json.load(file)

@app.route("/api", methods=["GET"])
def get_marks():
    names = request.args.getlist("name")  # Get all 'name' parameters
    marks = []

    for name in names:
        for student in data["students"]:
            if student["name"] == name:
                marks.append(student["marks"])

    return jsonify({"marks": marks})

# Vercel requires this
def handler(event, context):
    return app(event, context)