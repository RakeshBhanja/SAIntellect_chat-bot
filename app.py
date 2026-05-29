# app.py
from flask import Flask, request, jsonify, render_template
from brain import get_answer

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please type something!"}), 400
    reply = get_answer(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("=" * 45)
    print("  SAIntellect Bot is RUNNING!")
    print("  Open browser -> http://localhost:5000")
    print("=" * 45)
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000)
