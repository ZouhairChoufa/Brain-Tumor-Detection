from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# model = load_model('BrainTumors_model.h5')
model = load_model('models/BrainTumors_model.h5')
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Fonction pour vérifier l'extension de l'image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('page_home.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imag' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['imag']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)  # Sauvegarde le fichier dans le dossier
        except Exception as e:
            return jsonify({'error': f'File save error: {str(e)}'}), 500
            
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_labels[predicted_class[0]]
        print(f"Predicted Class: {predicted_label}")
        
        return jsonify({'result': predicted_label})
    
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)



