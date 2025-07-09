# driver-drowsiness

# Drowsiness Detection System

## Overview

This project implements a real-time drowsiness detection system using computer vision and deep learning. It monitors a user's eyes via a webcam feed and triggers an alarm if signs of drowsiness (prolonged eye closure) are detected. This system can be valuable for drivers, students, or anyone needing to maintain alertness.

## Features

* **Real-time Eye Monitoring:** Utilizes a webcam to continuously track eye movements.
* **Deep Learning Model:** Employs a Convolutional Neural Network (CNN) to classify eye states (open/closed).
* **Haar Cascade Classifiers:** Used for robust face and eye detection.
* **Drowsiness Score:** Accumulates a score based on continuous eye closure.
* **Audible Alarm:** Triggers an `alarm.wav` sound when the drowsiness score exceeds a predefined threshold.
* **Visual Alert:** Displays a red border around the video feed as drowsiness increases.
* **Cross-Platform (Python-based):** Can run on various operating systems where Python and its dependencies are supported.

## Technologies Used

* **Python 3.x**
* **OpenCV (`cv2`)**: For real-time video capture, image processing, and Haar cascade detection.
* **TensorFlow / Keras**: For loading and running the pre-trained CNN model.
* **NumPy**: For numerical operations, especially with image data.
* **Pygame Mixer**: For playing the alarm sound.

## Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vijay-0108/driver-drowsiness/tree/main
    cd drowsiness-detection-system
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install opencv-python tensorflow numpy pygame
    ```

4.  **Download Haar Cascade Files:**
    The project uses Haar cascade XML files for face and eye detection. These are typically found in the OpenCV installation directory or can be downloaded from the OpenCV GitHub repository.
    * Create a folder named `haar cascade files` in your project's root directory.
    * Place `haarcascade_frontalface_alt.xml`, `haarcascade_lefteye_2splits.xml`, and `haarcascade_righteye_2splits.xml` inside this folder.
    * You can often find them here: https://github.com/opencv/opencv/tree/master/data/haarcascades

5.  **Place the Pre-trained Model:**
    * Create a folder named `models` in your project's root directory.
    * Place your pre-trained Keras model file (`cnnCat4.h5`) inside this `models` folder.

6.  **Add the Alarm Sound:**
    * Place your alarm sound file (`alarm.wav`) in the root directory of the project.

## Usage

To run the drowsiness detection system, execute the main Python script:

```bash
python "drowsiness detection.py"
