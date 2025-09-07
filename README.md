# PennAiR Software Challenge - Shape Detection

## Overview

This project utilizes a computer vision algorithm to detect shapes on different backgrounds. It then traces their outlines, marks the center of each shape, and provides 3D coordinates.

## Implementation Summary

### Algorithm Approach
- **Edge-Based Consistency Detection** to identify uniform color regions
- **Multiple Stages of Morphological Opertations** to separate and clean shapes
- **Compactness-Based Filtering Criteria** to distinguish shapes from random noise
- **Camera Projection** to transform 2D coordinates to real-world 3D coordinates

## Parts Completed

### Part 1: Static Image Detection
- Successfully detects all 5 shapes on grass background
- Traces outlines and marks centers
- **Output:** Processed image found in `output\best_combo\annotated_PennAir 2024 App Static.png`

### Part 2: Video Processing
- Applies algorithm frame-by-frame to video stream
- Maintains consistent detection across frames
- **Output:** Annotated video found in `output\best_combo\annotated_PennAir 2024 App Dynamic.png`

### Part 3: Background Agnostic
- Needed to pivot on strategy, changing from value/brightness to edge-based consistency detection
- Pretty consistent ID of center, but outline does not completely match shape and sometimes ID's background as a shape
- Use of morphological operations drastically decreased error, but did not fully eliminate
- **Output:** Annotated video found in `output\best_combo\annotated_PennAir 2024 App Dynamic Hard.png`

### Part 4: 3D Positioning
- Calculates depth using circle's known radius
- Provides X, Y, Z coordinates relative to camera
- Uses camera intrinsic matrix to transform 2D coordinates to 3D

## Technical Approach

### Shape Detection Pipeline

#### Original Approach - Good for grassy background but not adaptable
1. **Preprocessing:** Convert the image to Hue-Saturation-Value format
2. **Value/Brightness:** Utilize value/brightness threshold to identify shapes
4. **Noise Removal:** Utilize erosion and compactness-filtering to more clearly identify shapes
5. **3D Calculation:** Identify most circular object for depth reference and use to calculate 3D coordinates for all objects

#### Reimagined Approach - Not as good as original for grass, but more consistent across variable backgrounds
1. **Preprocessing:** Convert the image to grayscale
2. **Edge Detection:** Use edge detection to locate general area of shapes
4. **Noise Removal:** Utilize erosion and compactness-filtering to more clearly identify shapes
5. **Redefine Contours:** Utilize approximation and convex hull to attempt to give back the original shape and size to the contour
6. **3D Calculation:** Identify most circular object for depth reference and use to calculate 3D coordinates for all objects

### Challenges and Difficulties
The initial approach originally worked well for both the static and dynamic situations on the grassy background, using a brightness threshold to separate the shapes from the background. However, the algorithm did not work too well for the dynamic hard situation. I attempted a few different methods with little success. Changing thresholds seemed semi-work, but I wanted an algorithm that could be applied to any background without manipulation. I thought about using the edge-detection to identify regions where colors were mostly uniform, as the background was noisy. I was a little surprised it worked, as some of the shapes in the dynamic hard situation were gradients, but the grayscale effect made them close enough that the tolerance given to the edge detection identified them as a uniform region. Seeing there was precedence to do so, I moved forward fine-tuning The biggest problem was that the noisy background in the dynamic hard situation ranged from white to black in the gray-scale format, so they were complete opposite. Using smoothing and erosion, as well as many other Morphological Operations, I was able to detach the shapes from any random noise and even fill in a lot of the gaps in the background. However, this was done at the cost of the shapes' original sizes and structure. I was able to undo much of the size alteration with dilation, but the shape was a bigger issue. In the end, I made the edges more rigid and using convex hull, but was not able to restore the full structure of the original shapes for the outlines. I was still pretty happy with the progress, as the algorithm now does a pretty good job at identifying the general location of the shapes on the screen despite the background.

## Installation & Usage

### Requirements
```bash
pip install opencv-python numpy
```

### Running the Code
```bash
# Save best 2D/3D annotated files for all 3 challenges to output/best_combo
python best_combo.py

# Save brightness-based detection 2D/3D annotated files for first 2 challenges to output/brightness
python brightness.py

# Save consistency-based detection 2D/3D annotated files for all 3 challenges to output/brightness
python consistency.py
```

### File Structure
```
PennAiR/
├── README.md
├── requirements.txt
├── data/
│   ├── PennAir 2024 App Static.png
│   ├── PennAir 2024 App Dynamic.mp4
│   └── PennAir 2024 App Dynamic Hard.mp4
├── output/
│   ├── best_combo
|   |   ├── annotated_3D_PennAir 2024 App Dynamic Hard.png
|   |   ├── annotated_3D_PennAir 2024 App Dynamic.png
|   |   ├── annotated_3D_PennAir 2024 App Static.png
|   |   ├── annotated_PennAir 2024 App Dynamic Hard.png
|   |   ├── annotated_PennAir 2024 App Dynamic.png
|   |   └── annotated_PennAir 2024 App Static.png
│   ├── brightness
|   |   ├── annotated_3D_PennAir 2024 App Static.png
|   |   ├── annotated_3D_PennAir 2024 App Dynamic.png
|   |   ├── annotated_PennAir 2024 App Static.png
|   |   ├── annotated_PennAir 2024 App Dynamic.png
│   └── consistency
|       ├── annotated_3D_PennAir 2024 App Dynamic Hard.png
|       ├── annotated_3D_PennAir 2024 App Dynamic.png
|       ├── annotated_3D_PennAir 2024 App Static.png
|       ├── annotated_PennAir 2024 App Dynamic Hard.png
|       ├── annotated_PennAir 2024 App Dynamic.png
|       └── annotated_PennAir 2024 App Static.png
├── best_combo.py
├── brightness.py
├── consistency.py
├── image_3D_brightness.ipynb
├── image_3D_consistency.ipynb
├── image_brightness.ipynb
├── image_consistency.ipynb
├── video_3D_agnostic_consistency.ipynb
├── video_3D_brightness.ipynb
├── video_3D_consistency.ipynb
├── video_agnostic_consistency.ipynb
├── video_brightness.ipynb
└── video_consistency.ipynb
```

## Results

### Performance Metrics
- **Detection Accuracy:** Successfully identifies all 5 shapes consistently
- **Background Robustness:** Works on both grass and mutli-color, noisy backgrounds

### Challenges Overcome
1. **Background Variation:** Developed edge-consistency detection instead of color-based thresholding
2. **Shape Separation:** Using erosion helped prevent shapes from being grouped in with random noise
3. **Compactness Filtering Criteria:** Compactness-based filtering helped remove random noise that was being identified as shapes

## Algorithm Innovations

### Edge Consistency Masking
Instead of regular color thresholding, uses edge detection to identify regions with consistent internal colors.

### Repeated Morphological Operations
Combines opening, closing, and targeted erosion in order to separate shapes that touch random noise.

### Compact-Based Filtering Criteria
One of the biggest problems with the agnostic background is that the algorithm was identifying patches of the background as shapes. These invalid shapes had large perimeter to area ratios, which was used to filter most of them out.

### Rebuilding Eroded Contours
Due to the immense erosion and smoothing needed to separate the shapes from noise, contours had to be rebuilt and made more rigid to attempt to regain their original structure.

## Video Demonstrations

Embedded videos showing algorithm performance:
- `annotated_PennAir 2024 App Static.png` - Annotated 2D Image
- <img width="1920" height="1080" alt="annotated_PennAir 2024 App Static" src="https://github.com/user-attachments/assets/dc11667d-dad8-4bfa-a003-ed6975d3ac36" />

- `annotated_3D_PennAir 2024 App Static.png` - Annotated 3D Image
- <img width="1920" height="1080" alt="annotated_3D_PennAir 2024 App Static" src="https://github.com/user-attachments/assets/4da81c33-14ae-4cbd-abac-43d0b4e41e02" />

- `annotated_PennAir 2024 App Dynamic.mp4` - Annotated 2D Video (Grassy Background)
- `annotated_3D_PennAir 2024 App Dynamic.mp4` - Annotated 3D Video (Grassy Background)
- `annotated_PennAir 2024 App Dynamic Hard.mp4` - Annotated 3D Video (Agnostic Background)
- `annotated_3D_PennAir 2024 App Dynamic Hard.mp4` - Annotated 3D Video (Agnostic Background)

## Future Improvements

If given additional time, potential enhancements would include:
- ROS2 implementation 
- Machine learning classifier for shape type identification to help rebuild contours
- Improvements to efficienncy
