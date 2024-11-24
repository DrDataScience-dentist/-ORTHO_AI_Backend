from flask import Flask, request, jsonify, send_file
import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths for models
MODEL_PATHS = {
    "frontal": "ORTHO-AI/FRONTAL AI/FRONTAL AI.pt",
    "right_lateral": "ORTHO-AI/LATERAL AI MODEL/LATERAL.pt",
    "left_lateral": "ORTHO-AI/LATERAL AI MODEL/LATERAL.pt",
    "upper_occlusal": "ORTHO-AI/OCCLUSAL AI/occlusal.pt",
    "lower_occlusal": "ORTHO-AI/OCCLUSAL AI/occlusal.pt"
}

OUTPUT_PDF = "Annotated_Orthodontic_Images.pdf"

# Generate a diagnosis based on predictions
def generate_diagnosis(predictions):
    features = set(predictions)
    diagnosis = "Orthodontic Diagnosis:\n"
    if "Class II" in features:
        diagnosis += "Class II Malocclusion detected.\n"
    elif "Class III" in features:
        diagnosis += "Class III Malocclusion detected.\n"
    else:
        diagnosis += "Class I Malocclusion detected.\n"
    if "Deep Bite" in features:
        diagnosis += "Deep Bite detected.\n"
    if "Open Bite" in features:
        diagnosis += "Open Bite detected.\n"
    if "Spacing" in features:
        diagnosis += "Spacing issues detected.\n"
    diagnosis += "\nRecommendation: Consult an orthodontist for detailed evaluation."
    return diagnosis

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({"message": f"File uploaded successfully: {file.filename}"}), 200

@app.route('/process', methods=['POST'])
def process_images():
    all_predictions = []
    with PdfPages(OUTPUT_PDF) as pdf:
        for image_type, model_path in MODEL_PATHS.items():
            image_path = os.path.join(UPLOAD_FOLDER, f"{image_type}.jpg")
            if not os.path.exists(image_path):
                continue
            model = YOLO(model_path)
            results = model.predict(source=image_path, conf=0.25)
            labels = results[0].names
            predictions = [labels[int(cls)] for cls in results[0].boxes.cls]
            all_predictions.extend(predictions)
            annotated_image = results[0].plot()
            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_image)
            plt.axis('off')
            plt.title(f"{image_type.capitalize()} Predictions")
            pdf.savefig()
            plt.close()
        diagnosis = generate_diagnosis(all_predictions)
        plt.figure(figsize=(10, 5))
        plt.axis('off')
        plt.text(0.5, 0.5, diagnosis, ha="center", va="center", fontsize=14)
        pdf.savefig()
        plt.close()
    return jsonify({"message": "Images processed and PDF generated."}), 200

@app.route('/report', methods=['GET'])
def download_report():
    if os.path.exists(OUTPUT_PDF):
        return send_file(OUTPUT_PDF, as_attachment=True)
    return jsonify({"error": "Report not found."}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
