import cv2
import numpy as np
import os

FILE_DIR = "data"
OUTPUT_DIR = os.path.join("output", "consistency")
FILE_NAMES = ["PennAir 2024 App Static.png", "PennAir 2024 App Static.png", "PennAir 2024 App Dynamic.mp4", "PennAir 2024 App Dynamic.mp4", "PennAir 2024 App Dynamic Hard.mp4", "PennAir 2024 App Dynamic Hard.mp4"]

# Intrinsic Matrix
K = np.array([[2564.3186869, 0, 0], [0, 2569.70273111, 0], [0, 0, 1]])

# Circle Radius
R = 10.0

# Helper Functions

def find_most_circular_contour(contours):

    best_circularity = 0
    best_contour = None
    best_index = -1
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0 and area > 500:  
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity:
                best_circularity = circularity
                best_contour = contour
                best_index = i
    
    return best_contour, best_circularity, best_index

def calculate_3d_coordinates(contours, centers, camera_matrix, circle_radius):

    coordinates_3d = []
    reference_depth = None
    
    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Find the most circular contour
    circle_contour, circularity, circle_index = find_most_circular_contour(contours)
    
    if circle_contour is not None and circularity > 0.8:

        # Calculate depth from the most circular object
        area = cv2.contourArea(circle_contour)
        radius_pixels = np.sqrt(area / np.pi)
        
        # Calculate reference depth
        reference_depth = (fx * circle_radius) / radius_pixels
            
    else:

        # Default depth
        reference_depth = 100.0 
    
    # Calculate 3D coordinates for all shapes using reference depth
    for i, (contour, center) in enumerate(zip(contours, centers)):

        # Image coordinates
        u, v = center  
        
        # Calculate 3D coordinates
        Z = reference_depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        coordinates_3d.append((X, Y, Z))
    
    return coordinates_3d, circle_index

def draw_3d_annotations(frame, contours, centers, coordinates_3d, circle_index=-1):
    
    result_frame = frame.copy()
    
    # Draw contours
    cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 2)
    
    for i, (contour, center, coord_3d) in enumerate(zip(contours, centers, coordinates_3d)):
        cx, cy = center
        X, Y, Z = coord_3d
        
        # Get bounding box for text placement
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw center
        cv2.circle(result_frame, center, 5, (255, 255, 255), -1)

        # Place 3D coordinates
        text_3d = f"({X:.1f},{Y:.1f},{Z:.1f})"
        cv2.putText(result_frame, text_3d, (cx - 80, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result_frame

def edge_consistency_mask(gray):
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Invert 
    consistent_mask = cv2.bitwise_not(edge_dilated)
    
    return consistent_mask

def remove_noise_and_erode(noise_mask):

    # Removes noise
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    no_noise_mask = cv2.morphologyEx(noise_mask, cv2.MORPH_OPEN, noise_kernel, iterations=1)

    # Erode to separate shapes connected to random blotches
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded_mask = cv2.erode(no_noise_mask, erosion_kernel, iterations=2)

    return eroded_mask

def smooth_mask_and_fill_gaps(unsmoothed_mask):
    
    smoothing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Fill the gaps and smooth the edges
    filled_gaps_mask = cv2.morphologyEx(unsmoothed_mask, cv2.MORPH_CLOSE, smoothing_kernel)

    # Smooth more with blur
    smoothed_mask = cv2.medianBlur(filled_gaps_mask, 15)

    return smoothed_mask

def heavy_erode(non_eroded_mask):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Remove small noise while keeping size of surviving shapes
    opened_mask = cv2.morphologyEx(non_eroded_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    # Heavy erosion to separate shapes
    separated_mask = cv2.erode(opened_mask, kernel, iterations=7)

    return separated_mask

def filter_countours_compactness(old_mask, contours, min_area=1845, max_compactness=55, min_perimeter=0):

    # Create empty mask from old mask shape
    new_mask = np.zeros_like(old_mask)

    for contour in contours:

        # Get area/perimeter for each contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Filter by area and compactness of shape
        if area > min_area and perimeter > min_perimeter and (perimeter * perimeter) / area < max_compactness:
            # Draw passing contours on new mask
            cv2.fillPoly(new_mask, [contour], 255)
    
    return new_mask

def dilate_to_undo_erosion(clean_mask):

    # Undo erosion that filled gaps
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final_mask = cv2.dilate(clean_mask, erosion_kernel, iterations=2)

    # Undo heavy erosion that separated shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    final_mask = cv2.dilate(final_mask, kernel, iterations=7)

    return final_mask

def get_rigid_contours(contours, epsilon_size=0.01):

    rigid_contours = []

    for contour in contours:
        
        # Approximate contour with fewer points
        epsilon = epsilon_size * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        rigid_contours.append(approx)
    
    return rigid_contours

def get_hull_contours(contours):

    hull_contours = []
    
    for contour in contours:
        
        # Get convex hull of contour
        hull = cv2.convexHull(contour)
        hull_contours.append(hull)
    
    return hull_contours

def filter_noise_contours(contours, min_area=1000):

    # Remove noise
    filtered_contours = []

    for contour in contours:
        
        area = cv2.contourArea(contour)

        # How much noise to filter
        if area > 1000:
            filtered_contours.append(contour)
    
    return filtered_contours

def get_centers(contours):

    # Calculate centers of contours
    centers = []

    for contour in contours:

        # Calculate moments
        M = cv2.moments(contour)
        
        if M["m00"] != 0: 

            # x coord
            cx = int(M["m10"] / M["m00"])

            # y coord
            cy = int(M["m01"] / M["m00"])

            centers.append((cx, cy))
    
    return centers

# Main Functions

def image_consistency(file_name, show_steps=False):

    file = os.path.join(FILE_DIR, file_name)

    output_name = f"annotated_{file_name}"
    output = os.path.join(OUTPUT_DIR, output_name)

    # Load image
    img = cv2.imread(file)

    img = cv2.resize(img, (1920, 1080))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Consistency detection to create mask
    consistency_mask = edge_consistency_mask(gray)

    # Remove noise and erode
    cleaned_mask = remove_noise_and_erode(consistency_mask)

    # Smooth
    smooth_mask = smooth_mask_and_fill_gaps(cleaned_mask)

    # Erode
    eroded_mask = heavy_erode(smooth_mask)

    # Find contours
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by compactness
    clean_mask = filter_countours_compactness(eroded_mask, contours)

    # Undo erosion
    final_mask = dilate_to_undo_erosion(clean_mask)

    # Find/Draw Outlines and Centers
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make contours more rigid
    rigid_contours = get_rigid_contours(contours)

    # Make contours convex
    hull_contours = get_hull_contours(rigid_contours)

    filtered_contours = filter_noise_contours(hull_contours)

    centers = get_centers(filtered_contours)

    # Copy original
    result = img.copy()

    # Draw outlines of shapes
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    # Draw centers and place text
    for i, (contour, center) in enumerate(zip(filtered_contours, centers)):
            cx, cy = center
            
            # Get bounding box for text placement
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw center
            cv2.circle(result, center, 5, (255, 255, 255), -1)
            
            # Place text
            text = f"({cx},{cy})"
            text_x = cx - 80
            text_y = y + h + 40
            cv2.putText(result, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if show_steps:
        # Display results
        cv2.imshow("Image Results", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.imwrite(output, result)

    cv2.imwrite(output, result)

    return result

def image_3D_consistency(file_name, show_steps=False):

    file = os.path.join(FILE_DIR, file_name)
    output_name = f"annotated_3D_{file_name}"
    output = os.path.join(OUTPUT_DIR, output_name)

    # Load image
    img = cv2.imread(file)

    img = cv2.resize(img, (1920, 1080))

    width = img.shape[1]
    height = img.shape[0]

    # Update camera matrix principal point if not provided
    # (Usually cx = width/2, cy = height/2)
    K_updated = K.copy()
    if K_updated[0, 2] == 0:  # If cx not set
        K_updated[0, 2] = width / 2
    if K_updated[1, 2] == 0:  # If cy not set
        K_updated[1, 2] = height / 2

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Consistency detection to create mask
    consistency_mask = edge_consistency_mask(gray)

    # Remove noise and erode
    cleaned_mask = remove_noise_and_erode(consistency_mask)

    # Smooth
    smooth_mask = smooth_mask_and_fill_gaps(cleaned_mask)

    # Erode
    eroded_mask = heavy_erode(smooth_mask)

    # Find contours
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by compactness
    clean_mask = filter_countours_compactness(eroded_mask, contours)

    # Undo erosion
    final_mask = dilate_to_undo_erosion(clean_mask)

    # Find/Draw Outlines and Centers
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make contours more rigid
    rigid_contours = get_rigid_contours(contours)

    # Make contours convex
    hull_contours = get_hull_contours(rigid_contours)

    filtered_contours = filter_noise_contours(hull_contours)

    centers = get_centers(filtered_contours)

    # Copy original
    result = img.copy()

    # Draw outlines of shapes
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    # Calculate 3D coordinates and get index in list of contours of most circular object
    coordinates_3d, circle_index = calculate_3d_coordinates(filtered_contours, centers, K_updated, R)

    # Draw results with 3D annotations
    result = draw_3d_annotations(result, filtered_contours, centers, coordinates_3d, circle_index)

    if show_steps:
        # Display results
        cv2.imshow("Image Results", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    cv2.imwrite(output, result)

    return result

def video_consistency(file_name, show_steps=False):

    file = os.path.join(FILE_DIR, file_name)
    output_name = f"annotated_{file_name}"
    output = os.path.join(OUTPUT_DIR, output_name)

    vid = cv2.VideoCapture(file)

    # Video properties
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer to save output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Loop through video frames
    while True:
        ret, frame = vid.read()

        # Break if no frame is returned
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Consistency detection to create mask
        consistency_mask = edge_consistency_mask(gray)

        # Remove noise and erode
        cleaned_mask = remove_noise_and_erode(consistency_mask)

        # Smooth
        smooth_mask = smooth_mask_and_fill_gaps(cleaned_mask)

        # Erode
        eroded_mask = heavy_erode(smooth_mask)
        
        # Find contours
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by compactness
        clean_mask = filter_countours_compactness(eroded_mask, contours)

        # Undo erosion
        final_mask = dilate_to_undo_erosion(clean_mask)

        # Find/Draw Outlines and Centers
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Make contours more rigid
        rigid_contours = get_rigid_contours(contours)

        # Make contours convex
        hull_contours = get_hull_contours(rigid_contours)

        filtered_contours = filter_noise_contours(hull_contours)
        
        centers = get_centers(filtered_contours)

        # Copy original frame
        result_frame = frame.copy()

        # Draw contours
        cv2.drawContours(result_frame, filtered_contours, -1, (10, 10, 10), 2)

        # Draw centers
        for center in centers:
            cv2.circle(result_frame, center, 5, (10, 10, 10), -1)
        
        # Write output
        out_vid.write(result_frame)
        
        if show_steps:
            # Display results
            cv2.imshow('Video Results', result_frame)
            if cv2.waitKey(1) & 0xFF == 13:
                break

    vid.release()
    out_vid.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def video_3D_consistency(file_name, show_steps=False):

    file = os.path.join(FILE_DIR, file_name)
    output_name = f"annotated_3D_{file_name}"
    output = os.path.join(OUTPUT_DIR, output_name)

    vid = cv2.VideoCapture(file)

    # Video properties
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer to save output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Update camera matrix principal point if not provided
    # (Usually cx = width/2, cy = height/2)
    K_updated = K.copy()
    if K_updated[0, 2] == 0:  # If cx not set
        K_updated[0, 2] = width / 2
    if K_updated[1, 2] == 0:  # If cy not set
        K_updated[1, 2] = height / 2

    # Loop through video frames
    # Loop through video frames
    while True:
        ret, frame = vid.read()

        # Break if no frame is returned
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Consistency detection to create mask
        consistency_mask = edge_consistency_mask(gray)

        # Remove noise and erode
        cleaned_mask = remove_noise_and_erode(consistency_mask)

        # Smooth
        smooth_mask = smooth_mask_and_fill_gaps(cleaned_mask)

        # Erode
        eroded_mask = heavy_erode(smooth_mask)
        
        # Find contours
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by compactness
        clean_mask = filter_countours_compactness(eroded_mask, contours)

        # Undo erosion
        final_mask = dilate_to_undo_erosion(clean_mask)

        # Find/Draw Outlines and Centers
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Make contours more rigid
        rigid_contours = get_rigid_contours(contours)

        # Make contours convex
        hull_contours = get_hull_contours(rigid_contours)

        filtered_contours = filter_noise_contours(hull_contours)
        
        centers = get_centers(filtered_contours)

        # Copy original frame
        result_frame = frame.copy()

        # Calculate 3D coordinates and get index in list of contours of most circular object
        coordinates_3d, circle_index = calculate_3d_coordinates(filtered_contours, centers, K_updated, R)
        
        # Draw results with 3D annotations
        result_frame = draw_3d_annotations(result_frame, filtered_contours, centers, coordinates_3d, circle_index)
        
        # Write output
        out_vid.write(result_frame)
        
        if show_steps:
            # Display results
            cv2.imshow('Video Results', result_frame)
            if cv2.waitKey(1) & 0xFF == 13:
                break

    vid.release()
    out_vid.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def video_agnostic_consistency(file_name, show_steps=False):

    file = os.path.join(FILE_DIR, file_name)
    output_name = f"annotated_{file_name}"
    output = os.path.join(OUTPUT_DIR, output_name)

    vid = cv2.VideoCapture(file)

    # Video properties
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer to save output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Loop through video frames
    while True:
        ret, frame = vid.read()

        # Break if no frame is returned
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Consistency detection to create mask
        consistency_mask = edge_consistency_mask(gray)

        # Remove noise and erode
        cleaned_mask = remove_noise_and_erode(consistency_mask)

        # Smooth
        smooth_mask = smooth_mask_and_fill_gaps(cleaned_mask)

        # Erode
        eroded_mask = heavy_erode(smooth_mask)
        
        # Find contours
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by compactness
        clean_mask = filter_countours_compactness(eroded_mask, contours)

        # Undo erosion
        final_mask = dilate_to_undo_erosion(clean_mask)

        # Find/Draw Outlines and Centers
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Make contours more rigid
        rigid_contours = get_rigid_contours(contours)

        # Make contours convex
        hull_contours = get_hull_contours(rigid_contours)

        filtered_contours = filter_noise_contours(hull_contours)
        
        centers = get_centers(filtered_contours)

        # Copy original frame
        result_frame = frame.copy()

        # Draw contours
        cv2.drawContours(result_frame, filtered_contours, -1, (0, 255, 0), 2)

        # Draw centers and coordinates
        for i, (contour, center) in enumerate(zip(filtered_contours, centers)):
            cx, cy = center
            
            # Get bounding box for text placement
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw center
            cv2.circle(result_frame, center, 5, (255, 255, 255), -1)
            
            # Place text
            text = f"({cx},{cy})"
            text_x = cx - 80
            text_y = y + h + 40
            cv2.putText(result_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Write output
        out_vid.write(result_frame)
        
        if show_steps:
            # Display results
            cv2.imshow('Video Results', result_frame)
            if cv2.waitKey(1) & 0xFF == 13:
                break

    vid.release()
    out_vid.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def video_3D_agnostic_consistency(file_name, show_steps=False):

    file = os.path.join(FILE_DIR, file_name)
    output_name = f"annotated_3D_{file_name}"
    output = os.path.join(OUTPUT_DIR, output_name)

    vid = cv2.VideoCapture(file)

    # Video properties
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer to save output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Update camera matrix principal point if not provided
    # (Usually cx = width/2, cy = height/2)
    K_updated = K.copy()
    if K_updated[0, 2] == 0:  # If cx not set
        K_updated[0, 2] = width / 2
    if K_updated[1, 2] == 0:  # If cy not set
        K_updated[1, 2] = height / 2

    # Loop through video frames
    while True:
        ret, frame = vid.read()

        # Break if no frame is returned
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Consistency detection to create mask
        consistency_mask = edge_consistency_mask(gray)

        # Remove noise and erode
        cleaned_mask = remove_noise_and_erode(consistency_mask)

        # Smooth
        smooth_mask = smooth_mask_and_fill_gaps(cleaned_mask)

        # Erode
        eroded_mask = heavy_erode(smooth_mask)
        
        # Find contours
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by compactness
        clean_mask = filter_countours_compactness(eroded_mask, contours)

        # Undo erosion
        final_mask = dilate_to_undo_erosion(clean_mask)

        # Find/Draw Outlines and Centers
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Make contours more rigid
        rigid_contours = get_rigid_contours(contours)

        # Make contours convex
        hull_contours = get_hull_contours(rigid_contours)

        filtered_contours = filter_noise_contours(hull_contours)
        
        centers = get_centers(filtered_contours)

        # Copy original frame
        result_frame = frame.copy()

        # Calculate 3D coordinates and get index in list of contours of most circular object
        coordinates_3d, circle_index = calculate_3d_coordinates(filtered_contours, centers, K_updated, R)
        
        # Draw results with 3D annotations
        result_frame = draw_3d_annotations(result_frame, filtered_contours, centers, coordinates_3d, circle_index)
        
        # Write output
        out_vid.write(result_frame)
        
        if show_steps:
            # Display results
            cv2.imshow('Video Results', result_frame)
            if cv2.waitKey(1) & 0xFF == 13:
                break

    vid.release()
    out_vid.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

func_names = [image_consistency, image_3D_consistency, video_consistency, video_3D_consistency, video_agnostic_consistency, video_3D_agnostic_consistency]

for file_name, func in zip(FILE_NAMES, func_names):
    print(f"Processing {file_name} with {func.__name__}...")
    func(file_name)
    print(f"Finished processing {file_name}.\n")