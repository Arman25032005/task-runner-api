from flask import Flask, request, jsonify
import subprocess
import os
import json
import sqlite3
import requests
import markdown
import re
from datetime import datetime
from collections import Counter
from bs4 import BeautifulSoup
from PIL import Image
import whisper
import pandas as pd
import numpy as np


app = Flask(__name__)
DATA_DIR = os.path.expanduser("~/data")

# Utility function to validate file paths
def validate_path(file_path):
    if not file_path.startswith(DATA_DIR):
        raise PermissionError("Access outside /data is restricted")
    return file_path

# Utility function to execute shell commands
def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return None, str(e)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Task Agent API!"})

@app.route("/task", methods=["POST"])
def run_task():
    data = request.get_json()
    if not data or "task" not in data:
        return jsonify({"error": "Task description is required"}), 400

    task = data["task"]
    structured_task = parse_task_with_llm(task)

    if not structured_task:
        return jsonify({"error": "Unable to understand task"}), 400

    try:
        output, error = execute_task(structured_task)
        if error:
            return jsonify({"error": error}), 500
        return jsonify({"message": output}), 200
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to call OpenAI API for task structuring
from dotenv import load_dotenv
import openai
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Set it in environment variables.")



def parse_task_with_llm(task):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that converts user task descriptions into structured commands."},
                {"role": "user", "content": task}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return None

# Function to execute specific tasks
def execute_task(task):
    if "install uv" in task:
        return execute_command("pip install uv")
    elif "run datagen.py" in task:
        email = extract_email_from_task(task)
        return execute_command(f"python3 datagen.py {email}")
    elif "format markdown" in task:
        return execute_command("npx prettier@3.4.2 --write /data/format.md")
    elif "count Wednesdays" in task:
        return count_weekdays("/data/dates.txt", "Wednesday")
    elif "sort contacts" in task:
        return sort_contacts_json("/data/contacts.json")
    elif "log files" in task:
        return extract_recent_logs("/data/logs/")
    elif "Markdown index" in task:
        return generate_markdown_index("/data/docs/")
    elif "fetch data from API" in task:
        return fetch_and_save_api_data()
    elif "clone git repo" in task:
        return clone_and_commit_repo()
    elif "run SQL query" in task:
        return execute_sql_query()
    elif "convert markdown to html" in task:
        return convert_markdown_to_html("/data/docs/example.md")
    elif "scrape website" in task:
        return scrape_website(task.split(" ")[-1])
    elif "compress image" in task:
        return compress_image("image.jpg", 50)
    elif "transcribe audio" in task:
        return transcribe_audio("audio.wav")
    elif "filter CSV" in task:
        return filter_csv("data.csv", {"column": "value"})
    elif "extract sender email" in task:
        return extract_sender_email("/data/email.txt")
    elif "extract credit card number" in task:
        return extract_credit_card_number("/data/credit-card.png")
    elif "find similar comments" in task:
        return find_most_similar_comments("/data/comments.txt")
    elif "calculate total sales for Gold tickets" in task:
        return calculate_total_sales_for_gold("/data/ticket-sales.db")
    else:
        return None, "Task not recognized"

# Supporting functions (same as original but included missing ones)
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def count_weekdays(file_path, weekday):
    validate_path(file_path)
    try:
        with open(file_path, "r") as f:
            dates = [line.strip() for line in f.readlines()]
        count = sum(1 for date in dates if datetime.strptime(date, "%Y-%m-%d").strftime("%A") == weekday)
        output_path = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path).split('.')[0]}-{weekday.lower()}.txt")
        with open(output_path, "w") as f:
            f.write(str(count))
        return f"Counted {count} {weekday}s", None
    except Exception as e:
        return None, str(e)

def sort_contacts_json(file_path):
    validate_path(file_path)
    try:
        with open(file_path, "r") as f:
            contacts = json.load(f)
        contacts.sort(key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))
        with open(file_path.replace(".json", "-sorted.json"), "w") as f:
            json.dump(contacts, f, indent=4)
        return "Contacts sorted successfully", None
    except Exception as e:
        return None, str(e)

def generate_markdown_index(directory):
    validate_path(directory)
    index = {}
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.startswith("# "):
                                index[file] = line.strip("# ").strip()
                                break
        with open(os.path.join(directory, "index.json"), "w") as f:
            json.dump(index, f, indent=4)
        return "Markdown index generated", None
    except Exception as e:
        return None, str(e)

def extract_recent_logs(directory):
    validate_path(directory)
    try:
        log_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".log")], key=os.path.getmtime, reverse=True)[:10]
        with open("/data/logs-recent.txt", "w") as f:
            for log_file in log_files:
                with open(log_file, "r") as lf:
                    first_line = lf.readline().strip()
                    f.write(first_line + "\n")
        return "Recent logs extracted", None
    except Exception as e:
        return None, str(e)

def extract_email_from_task(task):
    match = re.search(r'[\w\.-]+@[\w\.-]+', task)
    return match.group(0) if match else "unknown@example.com"

def fetch_and_save_api_data():
    response = requests.get("https://jsonplaceholder.typicode.com/posts")
    if response.status_code == 200:
        with open("/data/api_data.json", "w") as f:
            json.dump(response.json(), f, indent=4)
        return "API data saved successfully", None
    return None, "Failed to fetch API data"

def clone_and_commit_repo():
    repo_url = "https://github.com/example/repo.git"
    repo_dir = "/data/repo"
    execute_command(f"git clone {repo_url} {repo_dir}")
    execute_command(f"cd {repo_dir} && touch newfile.txt && git add . && git commit -m 'Automated commit'")
    return "Repo cloned and updated successfully", None

def convert_markdown_to_html(file_path):
    validate_path(file_path)
    try:
        with open(file_path, "r") as f:
            md_content = f.read()
        html_content = markdown.markdown(md_content)
        output_path = file_path.replace(".md", ".html")
        with open(output_path, "w") as f:
            f.write(html_content)
        return "Markdown converted to HTML", None
    except Exception as e:
        return None, str(e)
def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to fetch website"}
    
    soup = BeautifulSoup(response.text, "html.parser")
    return {"content": soup.prettify()[:500]}

def execute_sql_query(query, db_path="/data/ticket-sales.db"):
    validate_path(db_path)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()  # Fetch all results
        conn.close()
        
        # Save results to a file
        output_path = f"/data/sql_query_result.txt"
        with open(output_path, "w") as f:
            for row in result:
                f.write(str(row) + "\n")

        return f"Query executed successfully. Results saved to {output_path}", None
    except sqlite3.Error as e:
        return None, f"SQL error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def extract_email_from_task(task):
    match = re.search(r'[\w\.-]+@[\w\.-]+', task)
    return match.group(0) if match else "unknown@example.com"

def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to fetch website", None
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.prettify()[:500], None

def compress_image(filename, quality):
    image_path = os.path.join(DATA_DIR, filename)
    img = Image.open(image_path)
    compressed_path = image_path.replace(".", "_compressed.")
    img.save(compressed_path, "JPEG", quality=quality)
    return "Image compressed successfully", None

def transcribe_audio(filename):
    audio_path = os.path.join(DATA_DIR, filename)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"], None

def filter_csv(filename, filters):
    csv_path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(csv_path)
    for key, value in filters.items():
        df = df[df[key] == value]
    return df.to_json(orient="records"), None

def extract_sender_email(file_path):
    validate_path(file_path)
    try:
        with open(file_path, "r") as f:
            email_content = f.read()

        # Simple regex to match email addresses
        match = re.search(r"From:.*<([\w\.-]+@[\w\.-]+)>", email_content)
        sender_email = match.group(1) if match else "unknown@example.com"

        output_path = file_path.replace(".txt", "-sender.txt")
        with open(output_path, "w") as f:
            f.write(sender_email)
        
        return "Sender's email extracted successfully", None
    except Exception as e:
        return None, str(e)

def extract_credit_card_number(image_path):
    validate_path(image_path)
    try:
        try:
            import pytesseract
            if os.name == 'nt':  # Windows
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        except ImportError:
            return None, "pytesseract is not installed. Please install it using 'pip install pytesseract' and ensure Tesseract OCR is installed on your system"

        # Check if file exists
        if not os.path.exists(image_path):
            return None, f"Image file not found: {image_path}"

        # Check if file is an image
        try:
            img = Image.open(image_path)
        except Exception:
            return None, "File is not a valid image"

        # Preprocess image for better OCR (optional)
        img = img.convert('L')  # Convert to grayscale
        
        try:
            card_text = pytesseract.image_to_string(img)
        except pytesseract.TesseractError as e:
            return None, f"OCR Error: {str(e)}"

        # Extract the first sequence of 16 digits
        match = re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", card_text)
        if not match:
            return None, "No valid credit card number found in the image"
            
        card_number = match.group(0).replace(" ", "").replace("-", "")

        output_path = image_path.replace(".png", ".txt")
        with open(output_path, "w") as f:
            f.write(card_number)
        
        return "Credit card number extracted successfully", None
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def find_most_similar_comments(file_path):
    validate_path(file_path)
    try:
        # Import required dependencies
        import numpy as np
        from openai import OpenAI  # Updated import

        # Initialize OpenAI client
        client = OpenAI()  # Make sure OPENAI_API_KEY environment variable is set

        # Read comments
        with open(file_path, "r") as f:
            comments = [line.strip() for line in f.readlines()]
            
        if len(comments) < 2:
            return None, "Need at least 2 comments to find similar pairs"

        # Get embeddings for each comment
        try:
            embeddings = []
            for comment in comments:
                response = client.embeddings.create(
                    input=comment,
                    model="text-embedding-ada-002"
                )
                embeddings.append(response.data[0].embedding)
        except Exception as e:
            return None, f"Error getting embeddings: {str(e)}"

        def cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            return dot_product / (norm_v1 * norm_v2)

        # Find most similar pair
        max_similarity = -1
        most_similar_pair = (0, 1)  # Default to first two comments

        # Calculate pairwise similarities
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (i, j)

        # Write results to output file
        output_path = file_path.replace(".txt", "-similar.txt")
        with open(output_path, "w") as f:
            f.write(f"{comments[most_similar_pair[0]]}\n{comments[most_similar_pair[1]]}")
        
        return f"Most similar comments found with similarity score: {max_similarity:.4f}", None

    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def calculate_total_sales_for_gold(db_path="/data/ticket-sales.db"):
    validate_path(db_path)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type='Gold'")
        result = cursor.fetchone()[0]
        conn.close()

        output_path = db_path.replace(".db", "-gold-sales.txt")
        with open(output_path, "w") as f:
            f.write(str(result if result else 0))

        return "Total sales for Gold tickets calculated successfully", None
    except Exception as e:
        return None, str(e)


if __name__ == '__main__':
    app.run(debug=True)






        

