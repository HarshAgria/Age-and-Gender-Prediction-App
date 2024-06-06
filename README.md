# Age and Gender Detection Project

This project is a Python-based application for detecting the age and gender of individuals from live video feed using pre-trained deep learning models. The project uses OpenCV for video capture and image processing, and deep learning models for face detection, age prediction, and gender prediction.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- Detects faces in a live video feed.
- Predicts the age and gender of detected faces.
- Displays the predicted age and gender on the video feed.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/HarshAgria/Age-and-Gender-Prediction-app.git
    cd Age-and-Gender-Prediction-app
    ```

2. Install the required Python packages:
    ```sh
    pip install opencv-python opencv-python-headless
    ```

3. Download the pre-trained models and place them in the project directory:
    - [OpenCV Face Detector](https://github.com/spmallick/learnopencv/tree/master/AgeGender/opencv_face_detector.pbtxt)
    - [Age Detection Model](https://github.com/spmallick/learnopencv/tree/master/AgeGender/age_net.caffemodel)
    - [Gender Detection Model](https://github.com/spmallick/learnopencv/tree/master/AgeGender/gender_net.caffemodel)

## Usage

1. Run the script to start the age and gender detection:
    ```sh
    python age_gender_detection.py
    ```

2. A window will open displaying the video feed from your webcam with detected faces and predicted age and gender.

3. Press `q` to quit the application.

## Dependencies

- Python 3.x
- OpenCV
- Pre-trained deep learning models for face detection, age prediction, and gender prediction.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) for providing an open-source computer vision library.
- The authors of the pre-trained models used in this project.

