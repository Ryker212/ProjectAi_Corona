from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model and label encoder
loaded_model = joblib.load('../model/corona_classifier_model.pkl')
label_encoder = joblib.load('../model/label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูล JSON จากคำขอ
    data = request.get_json()
    
    # แปลงข้อมูลใหม่ให้เป็น DataFrame
    new_data = pd.DataFrame(data)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(new_data)
    
    # Get confidence scores for the predictions
    confidence_scores = loaded_model.predict_proba(new_data)

    # Convert predictions back to original labels
    predicted_labels = label_encoder.inverse_transform(predictions)

    # สร้างผลลัพธ์
    results = []
    for i, label in enumerate(predicted_labels):
        results.append({
            "prediction": label,
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)