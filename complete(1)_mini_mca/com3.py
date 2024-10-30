import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import os
import pytesseract
from PIL import Image
import io
import tensorflow as tf  # Ensure TensorFlow is imported
import numpy as np
import cv2  # Ensure OpenCV is imported

# Load the saved model for license plate detection
model_path = r'C:\Users\Emil\Desktop\complete(1)_mini_mca\model.h5'
model = tf.keras.models.load_model(model_path)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

UPLOAD_DIRECTORY = "uploaded_images"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# App layout
app.layout = html.Div([
    html.H1("License Plate Recognition", style={"textAlign": "center"}),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag an image here or ',
            html.A('upload a file')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),

    html.Div(id='upload-status', style={'margin-top': '10px', 'color': 'green'}),

    html.Div(id='output-image-upload', style={'margin-top': '20px'}),

    html.Button('Process Image', id='process-button', n_clicks=0),
    html.Div(id='extracted-text', style={'margin-top': '20px'})
])


# Function to predict license plate bounding box
def predict_license_plate(image):
    image_resized = cv2.resize(image, (224, 224)) / 255.0  # Resize and normalize
    image_resized = np.expand_dims(image_resized, axis=0)  # Expand dimensions for model input
    pred_bbox = model.predict(image_resized)
    return pred_bbox[0]


# Combined callback to display the uploaded image, detect license plate, crop, and extract text
@app.callback(
    [Output('upload-status', 'children'),
     Output('output-image-upload', 'children'),
     Output('extracted-text', 'children')],
    [Input('upload-image', 'contents'),
     Input('process-button', 'n_clicks')],
    [State('upload-image', 'filename')]
)
def process_image(contents, n_clicks, filename):
    if contents is not None:
        # Display the upload success message
        upload_message = f"File '{filename}' uploaded successfully!"

        # Decode the uploaded image
        data = contents.split(',')[1]
        image_data = base64.b64decode(data)
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if n_clicks > 0:
            # Predict bounding box
            pred_bbox = predict_license_plate(image)
            x_min, y_min, x_max, y_max = map(int, pred_bbox)
            height, width, _ = image.shape
            x_min = max(0, x_min) 
            y_min = max(0, y_min)
            x_max = min(width, x_max) + 250
            y_max = min(height, y_max)

            # Crop the image to the bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]

            # Extract text from the cropped license plate
            pil_image = Image.fromarray(cropped_image)
            extracted_text = pytesseract.image_to_string(pil_image)

            # Draw the bounding box on the original image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
            _, encoded_image = cv2.imencode('.jpg', image)
            processed_image = base64.b64encode(encoded_image).decode()

            # Encode cropped image for display
            _, cropped_encoded_image = cv2.imencode('.jpg', cropped_image)
            cropped_image_display = base64.b64encode(cropped_encoded_image).decode()

            # Display both the original image with the bounding box and the cropped image
            return (upload_message,
                    html.Div([
                        html.Img(src='data:image/jpeg;base64,{}'.format(processed_image),
                                 style={'width': '50%', 'margin': '20px auto'}),
                        html.P(f"Uploaded file: {filename}"),
                        html.H4("Cropped License Plate:"),
                        html.Img(src='data:image/jpeg;base64,{}'.format(cropped_image_display),
                                 style={'width': '30%', 'margin': '20px auto'})
                    ]),
                    html.Div([
                        html.H3("Extracted Text:"),
                        html.P(extracted_text)
                    ]))
        return upload_message, "", ""

    return "", "", ""


# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
