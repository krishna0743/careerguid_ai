from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from google import genai  # modern google-genai SDK
import time 

app = Flask(__name__)

# ---------- Configuration ----------
CSV_FILE = "data/career_dataset_cleaned.csv"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize client globally.
try:
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        # Rely on the execution environment to provide the key
        client = genai.Client()
except Exception as e:
    print(f"Failed to initialize Gemini Client: {e}")
    client = None

# ---------- Load Dataset ----------
try:
    # NOTE: Since the data file is not provided, we use a mock DataFrame to prevent crashes.
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        raise Exception("Empty DataFrame")
except Exception as e:
    print(f"Error loading CSV at {CSV_FILE}: {e}. Using a mock DataFrame.")
    # Mock data structure to ensure functions don't crash
    df = pd.DataFrame({
        "Career_category": ["Technology", "Healthcare", "Education"],
        "Skill": ["Programming, AI", "Empathy, Biology", "Teaching, Communication"],
        "Interests": ["Video Games, Robotics", "Patient Care, Research", "Mentoring, Learning"],
        "Hobby": ["Coding, Reading", "Volunteering, Fitness", "Tutoring, Public Speaking"],
        "Career": ["Software Engineer", "Nurse", "High School Teacher"],
        "Recommended_education": ["Bachelor's in CS", "BSN", "Master's in Education"]
    })


# ---------- Utility Functions (Unchanged for core logic) ----------
def get_categories():
    return sorted(df['Career_category'].dropna().unique().tolist())

def get_skills_for_category(category):
    rows = df[df['Career_category'].str.lower() == category.lower()]
    suggestions = []
    for idx, row in rows.iterrows():
        combined = []
        for col in ["Skill","Interests","Hobby"]:
            if pd.notna(row.get(col)):
                combined.append(str(row[col]))
        if combined:
            suggestions.append(", ".join(combined))
    return suggestions

def predict_career_from_skills(skills_input, count=1):
    skills_input_set = set([s.strip().lower() for s in skills_input.split(",") if s.strip()])
    scores = []
    for idx, row in df.iterrows():
        row_skills = set()
        for col in ["Skill","Interests","Hobby"]:
            if pd.notna(row.get(col)):
                row_skills.update([s.strip().lower() for s in str(row[col]).split(",") if s.strip()])
        overlap = len(skills_input_set & row_skills)
        if overlap > 0:
            scores.append((overlap, row))
    scores = sorted(scores, key=lambda x: -x[0])
    results = []
    for overlap, row in scores[:count]:
        results.append({
            "career": row.get("Career","N/A"),
            "category": row.get("Career_category","N/A"),
            "education": row.get("Recommended_education","N/A"),
            "score": overlap
        })
    return results

# ---------- Gemini Chatbot (FIXED) ----------
def chat_with_gemini(message):
    global client
    if not client:
        return "Error: Gemini client is not initialized. Check API key configuration."

    # Provide a system instruction to guide the AI's persona
    system_instruction = "You are an AI Career Counselor. Based on the user's message, provide supportive, insightful, and concise advice about job trends, required skills, or education paths in a friendly tone."
    
    # Implement simple exponential backoff for API robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # FIX 1: Use 'contents' instead of 'input'
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=message, 
                config={"system_instruction": system_instruction}
            )
            # FIX 2: Use standard response.text property
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Gemini chat error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
            else:
                print(f"Gemini chat error: {e}")
                return "Error contacting Gemini API after several retries."
    return "An unexpected error occurred."

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_categories_and_skills")
def get_categories_and_skills():
    categories = get_categories()
    known_skills = df[["Skill","Interests","Hobby"]].dropna().values.flatten()
    known_skills = [str(s).strip() for s in known_skills if str(s).strip()]
    return jsonify({"categories": categories, "known_skills": list(set(known_skills))})

@app.route("/get_category_rows")
def get_category_rows():
    category = request.args.get("category", "")
    rows = get_skills_for_category(category)
    return jsonify({"rows": rows})

@app.route("/predict-career", methods=["POST"])
def predict_career():
    data = request.get_json()
    skills_input = data.get("skills","")
    count = int(data.get("count",1))
    predictions = predict_career_from_skills(skills_input, count)
    return jsonify({"prediction": predictions})

@app.route("/analyze_resume", methods=["POST"])
def analyze_resume():
    data = request.get_json()
    resume_text = data.get("resume_text","").lower()
    extracted_skills = []
    
    all_skills_data = set()
    for col in ["Skill","Interests","Hobby"]:
        for item in df[col].dropna().values:
            all_skills_data.update([s.strip().lower() for s in str(item).split(",") if s.strip()])
            
    for skill in all_skills_data:
        if skill in resume_text and skill not in extracted_skills:
            extracted_skills.append(skill)
            
    career_pred = predict_career_from_skills(", ".join(extracted_skills), count=1)
    career_details = career_pred[0] if career_pred else {"career":"N/A","education":"N/A", "category": "N/A"}
    return jsonify({
        "extracted_skills": ", ".join(extracted_skills),
        "career_details": career_details
    })

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_msg = data.get("message","")
    reply = chat_with_gemini(user_msg)
    return jsonify({"response": reply})

# ---------- Run App ----------
if __name__ == "__main__":
    app.run(debug=True)
