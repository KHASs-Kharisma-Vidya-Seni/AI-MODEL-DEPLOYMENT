import dlib
from PIL import Image
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import numpy as np
from torchvision import transforms
import cv2, os, requests, dotenv, onnxruntime, torch

app = Flask(__name__)
CORS(app)
dotenv.load_dotenv()

# Import ML Token from IBM
api_ibm = os.getenv("API_KEY")

# Load face shape model
onnx_filename = './model/faceshapemodel.onnx'
ort_session = onnxruntime.InferenceSession(onnx_filename)

# Define haircut classifications for men and women
haircut_classification_men = {
    "Heart": ['slicked_back', 'taper_fade'],
    "Oval": ['curtain_haircut', 'edgar_cut', 'fluffy', 'layer_haircut', 'man_bun', 'mohawk_taper', 'morrissey', 'mullet', 'pompadour', 'side_parted_hair', 'spiky_hair', 'two_block'],
    "Oblong": ['caesar_cut', 'faux_hawk', 'french_crop', 'mandarin_haircut', 'textured_crop', 'tight_crop'],
    "Round": ['bowl_cut', 'cepmek', 'comb_over', 'comma_hair', 'paquito_haircut', 'undercut'],
    "Square": ['buzz_cut', 'fringe_haircut', 'quiff', 'taper_cut', 'the_flow_hairstyle'],
}
haircut_classification_women = {
    "Heart": ['bob_with_layer', 'curly_bob', 'feathery_fringe', 'middle_parted_bob', 'pixie_cut', 'retro_pixie', 'side_parted_lob'],
    "Oval": ['angled_bob', 'blunt_cut', 'bob_oval', 'boyfriend_bob', 'butterfly_haircut', 'choppy_bob', 'classic_bob', 'medium_layer', 'micro_bangs_with_see_through_bangs', 'shaggy', 'wavy_bob', 'wavy_hair_with_curtain_bangs', 'wolf_cut'],
    "Oblong": ['shaggy_bob'],
    "Round": ['hime_cut', 'long_layer_haircut', 'medium_wolf_cut'],
    "Square": ['french_bob'],
}

# Preprocess image gender
def preprocess_image_gender(img):
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (178, 218)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict gender
def predict_gender(img, header):
    payload_scoring = {"input_data": [{"values": img.tolist()}]}
    scoring_url = os.getenv("ENDPOINT_GENDER")
    response_scoring = requests.post(scoring_url, json=payload_scoring, headers=header)
    result = response_scoring.json()['predictions'][0]['values']
    gender = "Male" if result[0][1] > 0.5 else "Female"
    confidence = round(result[0][1] * 100, 2)
    return gender, confidence

# Process image shape
def preprocess_image_shape(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Predict face shape
def predict_face_shape(image):
    class_dict = {'Heart': 0, 'Oblong': 1, 'Oval': 2, 'Round': 3, 'Square': 4}
    output = ort_session.run(None, {'input': image.numpy()})
    probs = torch.softmax(torch.tensor(output[0]), dim=1)[0]
    confidence, predicted_idx = torch.max(probs, 0)
    predicted_class = next(key for key, value in class_dict.items() if value == predicted_idx.item())
    return predicted_class, confidence.item() * 100

# Face detection using dlib
detector = dlib.get_frontal_face_detector()

@app.route('/')
def index():
    return "Khass API is Online!!!"

@app.route('/hasil', methods=['POST'])
def hasil():
    if 'file' not in request.files:
        abort(400, "No file part")

    file = request.files['file']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    try:
        token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": api_ibm, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
        mltoken = token_response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        return jsonify({'error': "Tidak dapat terhubung ke IBM cloud (Unable to resolve host 'iam.cloud.ibm.com')"}), 500

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 0:
        abort(404, "No face detected")
    elif len(faces) > 1:
        abort(404, f"Multiple faces detected: {len(faces)}")

    face = faces[0]
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    width_increase = int((x2 - x1) * 0.4)
    height_increase = int((y2 - y1) * 0.4)
    
    x1 = max(0, x1 - width_increase)
    y1 = max(0, y1 - height_increase)
    x2 = min(image.shape[1], x2 + width_increase)
    y2 = min(image.shape[0], y2 + height_increase)

    cropped_image = image[y1:y2, x1:x2]
    gender_img = preprocess_image_gender(cropped_image)
    shape_img = preprocess_image_shape(Image.fromarray(cropped_image))

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
    gender, gender_confidence = predict_gender(gender_img, header)
    face_shape, face_shape_confidence = predict_face_shape(shape_img)

    if gender == "Male":
        recommended_haircuts = haircut_classification_men.get(face_shape, [])
    elif gender == "Female":
        recommended_haircuts = haircut_classification_women.get(face_shape, [])

    output = {
        "faceshape": face_shape,
        "faceshape_confidence": f"{face_shape_confidence:.2f}%",
        "gender": gender,
        "gender_confidence": f"{gender_confidence:.2f}%",
        "recommended_haircuts": recommended_haircuts
    }

    return jsonify(output)

@app.errorhandler(400)
@app.errorhandler(404)
def handle_error(error):
    error_msg = error.description if hasattr(error, 'description') else str(error)
    return jsonify({'error': error_msg}), error.code

if __name__ == '__main__':
    app.run(debug=True)
