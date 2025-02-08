# Road Lane Line Detection using CNN

## Overview
This project implements a **Road Lane Line Detection** system using a **Convolutional Neural Network (CNN)**. The goal is to detect lane lines in road images or video streams, which is a crucial component in autonomous driving and advanced driver assistance systems (ADAS).

## Features
- Uses **CNN** for accurate lane detection
- Supports both **image and video input**
- Preprocessing using edge detection and perspective transformation
- Post-processing with lane fitting
- Implemented in **Python** with **TensorFlow/Keras, OpenCV, and NumPy**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/road-lane-detection-cnn.git
   cd road-lane-detection-cnn
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The model is trained on road images with lane markings. You can use publicly available datasets such as:
- [Tusimple Lane Dataset](https://github.com/TuSimple/tusimple-benchmark)
- [CULane Dataset](https://xingangpan.github.io/projects/CULane.html)

## Model Architecture
The CNN model consists of:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for lane classification
- Output layer predicting lane boundaries

## Training
1. Prepare your dataset and organize it into `train` and `test` folders.
2. Run the training script:
   ```bash
   python train.py --epochs 10 --batch_size 32
   ```

## Testing
To test the trained model on an image or video:
```bash
python test.py --input path/to/image_or_video
```

## Results
The model outputs lane boundaries on the given input images or video frames. You can visualize the predictions using OpenCV.

## Example Usage
```python
from model import LaneDetector

detector = LaneDetector("model.h5")
result = detector.detect_lanes("test_image.jpg")
```

## Future Improvements
- Enhance model performance with more training data
- Implement real-time lane detection
- Improve robustness in different weather and lighting conditions

## Contact
For any questions or suggestions, contact **pjahanavi2811@example.com** or open an issue on GitHub.

