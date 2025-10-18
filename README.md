# Brain Tumor Detection Web Application

This project uses a Convolutional Neural Network (CNN) to classify brain tumors from MRI images. It includes the original Jupyter Notebook for model training and a Flask web application to provide a user-friendly interface for making predictions.

The model can identify four different classes: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**.

##  Web App Demo

![Web App Screenshot](image_eec381.png)

##  Model Performance

The model was trained for 15 epochs and achieved **98% accuracy** on the test set.

* **Training Performance:**
    ![Accuracy and Loss Plots](results/accuracy_loss_plots.png)

* **Classification Report:**
    ![Classification Report](results/classification_report.png)

##  How to Run

Follow these steps to run the web application on your local machine.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/Brain-Tumor-Detection.git](https://github.com/your-username/Brain-Tumor-Detection.git)
cd Brain-Tumor-Detection
```

**2. Create and activate a virtual environment:**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Flask application:**
```bash
flask run
```
Now, open your web browser and go to `http://12-7.0.0.1:5000` to use the application!

##  Project Structure
The project is organized to separate the machine learning model development from the web application deployment.

-   `app.py`: The main Flask application file.
-   `models/`: Contains the trained `.h5` model file.
-   `notebooks/`: Contains the Jupyter Notebook used for model training.
-   `results/`: Contains performance graphs from the model training.
-   `static/`: Contains CSS and image files for the web interface.
-   `templates/`: Contains the HTML files for the web interface.