from flask import Flask, render_template, request
import mlflow
from mlflow.tracking import MlflowClient
import os
import re
import string
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# ENSURE NLTK DATA (CRITICAL FIX)
# =========================
def ensure_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    try:
        WordNetLemmatizer().lemmatize("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

ensure_nltk()

# =========================
# TEXT PREPROCESSING
# =========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join(w for w in text.split() if w not in stop_words)

def removing_numbers(text):
    return "".join(c for c in text if not c.isdigit())

def lower_case(text):
    return " ".join(w.lower() for w in text.split())

def removing_punctuations(text):
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def removing_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def normalize_text(text):
    text = lower_case(text)
    text = removing_urls(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = lemmatization(text)
    return text

# =========================
# MLFLOW + DAGSHUB
# =========================
DAGSHUB_PAT = os.getenv("DAGSHUB_PAT")
if not DAGSHUB_PAT:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_PAT
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_PAT

mlflow.set_tracking_uri(
    "https://dagshub.com/sudarshansahane1044/emo_project.mlflow"
)

# =========================
# LOAD MODEL (PIPELINE)
# =========================
def get_model_uri(model_name="my_model"):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    for v in versions:
        if v.tags.get("stage") == "production":
            return f"models:/{model_name}/{v.version}"

    for v in versions:
        if v.tags.get("stage") == "staging":
            return f"models:/{model_name}/{v.version}"

    return f"models:/{model_name}/{versions[0].version}"

MODEL_NAME = "my_model"
MODEL_URI = get_model_uri(MODEL_NAME)
model = mlflow.pyfunc.load_model(MODEL_URI)

print(f"✅ Loaded pipeline model → {MODEL_URI}")

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form["text"]

        if not text.strip():
            return render_template("index.html", result="Please enter valid text")

        cleaned_text = normalize_text(text)
        prediction = model.predict([cleaned_text])[0]

        return render_template("index.html", result=prediction)

    except Exception as e:
        print("🔥 Prediction error:", e)
        raise

# =========================
# HEALTH CHECK
# =========================
@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI
    }

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
