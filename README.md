License Plate Detection using CNN
This project implements a binary image classifier using a Convolutional Neural Network (CNN) to detect the presence of license plates in images. It leverages TensorFlow/Keras, OpenCV, and Scikit-learn for data handling, model building, training, and evaluation.

🧠 Model Architecture
The CNN consists of:

3 convolutional layers with increasing filter sizes (32, 64, 128)

Max pooling after each convolutional block

Dropout layers for regularization

Fully connected dense layer

Sigmoid output layer for binary classification

📁 Folder Structure
bash
Copy
Edit
project-root/
│
├── PositiveImages_Dataset/
│   └── images/     # Images containing license plates
│
├── NegativeImages_Dataset/
│   └── images/     # Images without license plates
│
├── license_plate_detection_model.h5  # Trained model (generated after training)
├── main.py         # Main script
└── README.md       # Project documentation
🔧 Requirements
Make sure you are using Google Colab or have the following Python libraries installed:

bash
Copy
Edit
pip install tensorflow opencv-python-headless matplotlib scikit-learn
🚀 How to Run
Upload your image datasets under the respective folders:

/content/PositiveImages_Dataset/images/

/content/NegativeImages_Dataset/images/

Run the entire Python script (main.py) or the notebook cells in order.

📝 Workflow Steps
1. Data Preparation
Loads and preprocesses positive and negative images (resized to 128x128 and normalized).

Splits data into training and validation sets.

2. Model Development
Builds a CNN model using Keras' Sequential API.

3. Data Augmentation
Applies transformations like rotation, shifting, and horizontal flipping.

4. Model Training
Trains the model using augmented data for better generalization.

5. Training History Visualization
Plots training/validation accuracy and F1/precision/recall across epochs.

6. Model Evaluation
Displays a classification report on the validation dataset.

7. Model Testing
Predicts and visualizes the result for a custom test image.

📊 Sample Output
Validation Accuracy: ~90%+

Classification Report:

markdown
Copy
Edit
precision    recall  f1-score   support

         0       0.92      0.89      0.90        XX
         1       0.90      0.93      0.91        XX
🔍 Testing on a Custom Image
python
Copy
Edit
test_model_on_image(model, '/path/to/test_image.jpg')
The function displays the image and prints either:

"License Plate Detected"

"No License Plate Detected"

💾 Saving the Model
The trained model is saved as:

Copy
Edit
license_plate_detection_model.h5
You can load it later using:

python
Copy
Edit
from tensorflow.keras.models import load_model
model = load_model('license_plate_detection_model.h5')
📌 Notes
Ensure image folders are correctly placed before running.

Tune hyperparameters (epochs, batch size, augmentation settings) as needed.

You can improve results with more balanced and varied training data.

📧 Contact
For questions or collaboration, feel free to connect!
