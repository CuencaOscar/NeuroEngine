from flask import Flask, request
import requests
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'


@app.route('/api', methods=['POST'])
def api():
    content = request.form.get('image')
    output_directory = "predict_image/test"
    os.makedirs(output_directory, exist_ok=True)
    image_path = os.path.join(output_directory, 'image.jpg')
    response = requests.get(content)
    with open(image_path, 'wb') as f:
        f.write(response.content)
    model = tf.keras.models.load_model('./src/models/new_model.h5')
    class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    test_image = load_and_prepare_image(image_path)
    result = classify_image(test_image, model, class_names)
    return result

def load_and_prepare_image(img_path, target_size=(128, 128)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    img_array_expanded = tf.expand_dims(img, axis=0)
    return img_array_expanded

def classify_image(image, model, class_names):
    predictions = model.predict(image, verbose=0)
    scores = tf.nn.softmax(predictions[0])
    probabilities = {class_name: probability.numpy() * 100 for class_name, probability in zip(class_names, scores)}
    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
    formatted_probabilities = {k: f"{v:.2f}%" for k, v in sorted_probabilities.items()}
    highest_probability_class = max(formatted_probabilities, key=formatted_probabilities.get)
    highest_probability_value = formatted_probabilities[highest_probability_class]
    print('La info que mando es: ', {highest_probability_class: highest_probability_value})
    return {highest_probability_class: highest_probability_value}


@app.route('/')
def init():
    return "Tesis Prediction"


if __name__ == '__main__':
    app.run(debug=True)