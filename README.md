🧠 Parkinson's Disease Detection from Spiral & Wave Drawings
This project uses deep learning to detect Parkinson’s Disease from hand-drawn spiral and wave patterns — a clinically significant early sign of motor dysfunction. It includes a CNN model trained on augmented image datasets and a user-friendly Streamlit web app for real-time predictions.

📌 Project Highlights
🔍 Binary Classification: Healthy vs Parkinson.

🧠 Model: Custom CNN built using TensorFlow & Keras.

📈 Augmented Dataset: Spiral and wave drawings augmented with transformations.

💻 Web App: Upload an image and instantly get a prediction.

📊 Visual Analysis: Accuracy/Loss plots, Confusion Matrix, and classification report.

📂 Project Structure
yaml
Copy
Edit
├── Parkinson's_Disease_Detection.ipynb    # End-to-end model pipeline in notebook
├── parkinson_disease_detection.h5         # Trained Keras CNN model
├── parkinson's_disease_detection.py       # Python script for training & evaluation
├── dectapp.py                             # Streamlit frontend for predictions
├── train_set.npz                          # Preprocessed training dataset
├── test_set.npz                           # Preprocessed testing dataset
├── test_image_healthy.png                 # Sample healthy spiral drawing
├── test_image_parkinson.png               # Sample parkinsonian spiral drawing
🧪 Model Architecture
A custom CNN model was created with:

Convolution Layers: 4 layers with increasing filter sizes (128 → 64 → 32 → 32)

Pooling Layers: MaxPooling after each Conv layer

Dropout: To reduce overfitting

Dense Layers: Final Softmax layer for binary classification

📊 Results
Achieved high accuracy (~96%) on test set

Strong generalization due to data augmentation

Clear confusion matrix and classification report

Visual Comparison of Input Samples:

Healthy Drawing	Parkinsonian Drawing
	

🚀 How to Run Locally
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/parkinson-detection-drawing.git
cd parkinson-detection-drawing
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
<details> <summary>📦 Click to view main dependencies</summary>
streamlit

tensorflow

opencv-python

scikit-learn

matplotlib

seaborn

</details>
3. Launch the Streamlit App
bash
Copy
Edit
streamlit run dectapp.py
🖼️ Sample Predictions (from App)
Upload a drawing (spiral or wave) and see predictions in real time.

<img src="https://via.placeholder.com/400x250?text=Upload+Page+UI" alt="App UI"/>
🧠 Dataset
Collected spiral and wave images from clinical references

Images are resized, normalized, and augmented (rotation, flips)

Labels encoded using LabelEncoder and one-hot encoded for training

📈 Training Pipeline
Data loading from .npz files

Augmentation using ImageDataGenerator

Preprocessing: resize (128x128), grayscale conversion

CNN training for 70 epochs with Adam optimizer

Evaluation and visualization

🏁 Future Work
Expand dataset with more clinically verified samples

Incorporate tremor severity grading

Real-time sketching pad integration

📃 License
This project is open-source and free to use under the MIT License.

🙌 Acknowledgements
Spiral drawings are clinically proven indicators of motor control deterioration in Parkinson’s Disease.

This project is inspired by real-world applications of computer vision in healthcare.

