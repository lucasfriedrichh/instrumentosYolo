import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image, ImageDraw
from ultralytics import YOLO
import io
import base64
from datetime import datetime

model = YOLO('E:/Downloads/algoritmos_datasets/FlaskModelo/flask_modelo/train18/weights/best.pt')

def predict(image_path, IMG_SIZE=640):
    with Image.open(image_path) as image:
        # Ensure the image has 3 color channels (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((IMG_SIZE, IMG_SIZE))

       
        results = model.predict(image, show=True, show_labels=False, save_txt=True, conf = 0.5)

        predicted_images = []  # Create a list to store the predicted images

        for r in results:
            for detection in r:
                classe = detection.boxes.cls[0]
                acuracia = detection.boxes.conf[0]
                nump = detection.boxes.numpy().xywhn[0]

                im_array = detection.plot()  # Plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

                #im.show(im)

                # Convert the PIL image to bytes
                with io.BytesIO() as output:
                    im.save(output, format="PNG")
                    image_bytes = output.getvalue()

                image_base64 = base64.b64encode(image_bytes).decode('ascii')

                nump_list = nump.tolist()
                class_name = model.names[classe.item()]

                predictions = {
                    "classe": classe.item(),
                    "acuracia": acuracia.item(),
                    "coordenadas": nump_list,
                    "nome_da_classe": class_name,
                    "image_base64": image_base64  # Add the base64 encoded image to the predictions
                }

                predicted_images.append(predictions)

        return predicted_images

app = Flask(__name__)  # Create a Flask app instance
CORS(app)  # Add CORS to your app

@app.route('/', methods=['POST'])
def predict_request():
    try:
        # Get file and save it
        file = request.files.get('image')
        filename = secure_filename(file.filename)
        file.save(filename)

        # Resize the image to 640x640
        resized_filename = 'resized_image.png'
        with Image.open(filename) as original_image:
            resized_image = original_image.resize((640, 640))
            resized_image.save(resized_filename, format="PNG")

        predictions = predict(resized_filename)

        if not predictions:
            return jsonify({"error": "No objects found in the image"})

        # Create a directory for the text files
        output_folder = 'output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Create a text file for each prediction
        for idx, prediction in enumerate(predictions):
            txt_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_prediction_{idx + 1}.txt")
            with open(txt_filename, 'w') as txt_file:
                txt_file.write(f"{int(prediction['classe'])} {' '.join(map(str, prediction['coordenadas']))}\n")

        filtered_predictions = []

        for prediction in predictions:
            filtered_prediction = {
                "classe": prediction["classe"],
                "acuracia": round(prediction["acuracia"], 2),
                "coordenadas": prediction["coordenadas"],
                "nome_da_classe": prediction["nome_da_classe"],
            }
            filtered_predictions.append(filtered_prediction)

        response = jsonify({
            "predict": filtered_predictions
        })

        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        print('response::::')
        print(response.data)
        return response

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/batch/<path:folder_path>', methods=['GET'])
def batch_predict_request(folder_path):
    try:
        last_char = os.path.basename(os.path.normpath(folder_path))

        folder_path = folder_path.replace('\\', '/')

        
       

        # Create a directory for the text files
        output_folder = 'output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_folder_name = folder_path.split("output/", -1)
        all_predictions = []

        # Recursively traverse the directory tree
        for root, _, files in os.walk(folder_path):
            for image_file in files:
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, image_file)
                    predictions = predict(image_path)
                    classe = os.path.basename(os.path.normpath(root))

                    # Use relative path for the text file
                    relative_path = os.path.relpath(image_path, folder_path)
                    txt_filename = os.path.join(output_folder, f"{os.path.splitext(relative_path)[0]}.txt")

                    # Ensure the directory for the text file exists
                    os.makedirs(os.path.dirname(txt_filename), exist_ok=True)

                    with open(txt_filename, 'w') as txt_file:
                        for prediction in predictions:
                            txt_file.write(f"{classe} {' '.join(map(str, prediction['coordenadas']))}\n")

                    all_predictions.append({
                        "image_file": relative_path.replace('\\', '/'),
                        "predictions": predictions
                    })

        return jsonify({
            "batch_predictions": all_predictions
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Set limit to 32 MB (or adjust as needed)
    app.run(host='0.0.0.0', port=8083)
