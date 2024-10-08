from flask import Flask, request, jsonify
from flask_cors import CORS  # นำเข้า CORS
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the model and label encoder
loaded_model = joblib.load('../model/corona_classifier_model.pkl')
label_encoder = joblib.load('../model/label_encoder.pkl')

@app.route('/prediction', methods=['POST'])
def predict():

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Create DataFrame
        new_data = pd.DataFrame(data)

        # Convert 'true'/'false' strings to booleans for relevant columns
        bool_columns = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']
        for col in bool_columns:
            if col in new_data.columns:
                new_data[col] = new_data[col].map(lambda x: True if x == 'true' else (False if x == 'false' else x))

        # Check columns
        expected_columns = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']
        if not all(col in new_data.columns for col in expected_columns):
            return jsonify({'error': 'Input data does not have the required features'}), 400

        # Convert boolean columns to int
        new_data[expected_columns] = new_data[expected_columns].astype(int)

        # Make predictions
        predictions = loaded_model.predict(new_data)

        # รับคะแนนความเชื่อมั่น
        confidence_scores = loaded_model.predict_proba(new_data)

        # แปลงผลลัพธ์ที่ทำนายกลับไปเป็น label เดิม
        predicted_labels = label_encoder.inverse_transform(predictions)

        # สร้างผลลัพธ์
        results = []
        for i, label in enumerate(predicted_labels):
            results.append({
                "prediction": label,
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
