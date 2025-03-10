from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("rf_llm_selector.pkl")

feature_columns = ["task_complexity", "data_type_text", "data_type_table", 
                   "format_requirement", "length_constraint"]

llm_mapping = {0: "GPT-4", 1: "Claude", 2: "Llama", 3: "Gemini", 4: "Mistral", 5: "Cohere"}
feedback_data = []
llm_explanations = {
    "GPT-4": "Best for complex reasoning and long documents.",
    "Claude": "Handles very long contexts efficiently.",
    "Llama": "Good for short texts and low-cost inference.",
    "Gemini": "Well-optimized for diverse tasks with large-scale data.",
    "Mistral": "Balances efficiency and performance for structured data.",
    "Cohere": "Ideal for short-form text generation with fast response time."
}
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([
            data["task_complexity"],
            data["data_type_text"],
            data["data_type_table"],
            data["format_requirement"],
            data["length_constraint"]
        ]).reshape(1, -1)


        prediction = model.predict(features)[0]
        confidence = np.max(model.predict_proba(features))

        recommended_llm = llm_mapping.get(prediction, "Unknown")

        feature_importance = model.feature_importances_
        important_features = {
            feature_columns[i]: feature_importance[i] for i in range(len(feature_columns))
        }

        explanation = llm_explanations.get(recommended_llm, "Chosen based on model training.")

        return jsonify({
            "recommended_llm": recommended_llm,
            "confidence_score": confidence,
            "reason": explanation,
            "feature_importance": important_features
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/feedback", methods=["POST"])
def collect_feedback():
    try:
        feedback = request.json
        feedback_data.append(feedback)  
        
        df = pd.DataFrame([feedback])
        df.to_csv("user_feedback.csv", mode="a", header=False, index=False)

        return jsonify({"message": "Feedback received successfully!"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
