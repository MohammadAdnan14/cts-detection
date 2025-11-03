import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import skeletonize
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

def advanced_preprocessing(img):
    """Enhanced preprocessing optimized for real dental X-rays."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Bilateral filter preserves edges better than Gaussian
    denoised1 = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # More aggressive CLAHE for low-contrast X-rays
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised1)
    
    # Advanced denoising (Non-local Means)
    denoised2 = cv2.fastNlMeansDenoising(enhanced, None, h=12, 
                                         templateWindowSize=7, 
                                         searchWindowSize=21)
    
    # Sharpen to make cracks more visible
    kernel_sharpen = np.array([[-1,-1,-1], 
                               [-1, 9,-1], 
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised2, -1, kernel_sharpen)
    
    return gray, enhanced, denoised2, sharpened

def segment_teeth_simple_geometric(img):
    """Simplest and most reliable - just use geometric region for panoramic X-rays."""
    h, w = img.shape
    
    # Create mask for dental arch - the MIDDLE HORIZONTAL BAND
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # ADJUSTED FOR PANORAMIC X-RAYS
    y_top = int(h * 0.35)      # Start from 35% (includes upper teeth)
    y_bottom = int(h * 0.65)   # End at 65% (includes lower teeth)
    x_left = int(w * 0.08)     # Left boundary (exclude skull edge)
    x_right = int(w * 0.92)    # Right boundary (exclude skull edge)
    
    # Fill the rectangular region
    mask[y_top:y_bottom, x_left:x_right] = 255
    
    # Optional: add slight intensity filter to remove very dark background
    _, bright = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)  # Very low threshold
    mask = cv2.bitwise_and(mask, bright)
    
    # Light morphology to smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply mask
    roi = cv2.bitwise_and(img, img, mask=mask)
    
    return roi, mask

def segment_teeth_adaptive(img):
    """Adaptive approach - uses image statistics to find teeth."""
    h, w = img.shape
    
    # Calculate intensity statistics
    non_zero = img[img > 10]  # Exclude pure black background
    if len(non_zero) > 0:
        mean_intensity = np.mean(non_zero)
        std_intensity = np.std(non_zero)
        
        # Teeth are typically 1 std above mean in panoramic X-rays
        tooth_threshold = mean_intensity + 0.3 * std_intensity
    else:
        tooth_threshold = 100
    
    # Create mask based on adaptive threshold
    _, adaptive_mask = cv2.threshold(img, int(tooth_threshold), 255, cv2.THRESH_BINARY)
    
    # Still use geometric mask to exclude far edges
    geo_mask = np.zeros((h, w), dtype=np.uint8)
    
    # More generous boundaries
    y_start = int(h * 0.20)  # More relaxed
    y_end = int(h * 0.80)
    x_start = int(w * 0.05)
    x_end = int(w * 0.95)
    
    geo_mask[y_start:y_end, x_start:x_end] = 255
    
    # Combine
    final_mask = cv2.bitwise_and(adaptive_mask, geo_mask)
    
    # Morphological refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply
    roi = cv2.bitwise_and(img, img, mask=final_mask)
    
    return roi, final_mask

def segment_teeth_region_balanced(img):
    """Balanced approach - segment teeth area WITHOUT being too strict."""
    h, w = img.shape
    
    # Create a mask for DENTAL ARCH region (where teeth are located)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define dental arch region (approximate, generous)
    y_start = int(h * 0.25)
    y_end = int(h * 0.75)
    x_start = int(w * 0.10)
    x_end = int(w * 0.90)
    
    # Create rectangular region (dental arch area)
    mask[y_start:y_end, x_start:x_end] = 255
    
    # Additional refinement: keep brighter regions (teeth are bright)
    # But use LOWER threshold so we don't lose teeth
    _, bright_mask = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)  # Relaxed threshold!
    
    # Combine geometric + intensity masks
    combined_mask = cv2.bitwise_and(mask, bright_mask)
    
    # Light morphology - just to clean up noise, NOT to remove teeth
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours and filter ONLY very large (skull edges) and very small (noise)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    refined_mask = np.zeros_like(img)
    
    if contours:
        valid_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # VERY RELAXED filters:
            # Remove only: 
            # - Tiny noise (< 200 pixels)
            # - HUGE skull regions (> 30000 pixels)
            
            if 200 < area < 30000:
                valid_regions.append(contour)
        
        if valid_regions:
            cv2.drawContours(refined_mask, valid_regions, -1, 255, -1)
            
            # Expand mask slightly to include tooth edges
            kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            refined_mask = cv2.dilate(refined_mask, kernel_expand, iterations=1)
        else:
            # If no regions pass filter, use the combined mask directly
            refined_mask = combined_mask
    else:
        # Fallback: use geometric + intensity mask
        refined_mask = combined_mask
    
    # Apply mask
    roi = cv2.bitwise_and(img, img, mask=refined_mask)
    
    return roi, refined_mask

def detect_cracks_advanced(roi_img):
    """Multi-method edge detection optimized for crack detection."""
    # Multiple Canny passes with different parameters
    blurred1 = cv2.GaussianBlur(roi_img, (3, 3), 0.8)
    blurred2 = cv2.GaussianBlur(roi_img, (5, 5), 1.4)
    
    median = np.median(roi_img[roi_img > 0]) if np.any(roi_img > 0) else 128
    
    # Fine edges (for thin cracks)
    edges_fine = cv2.Canny(blurred1, int(0.2*median), int(0.8*median))
    
    # Medium edges
    edges_medium = cv2.Canny(blurred2, int(0.4*median), int(1.2*median))
    
    # Coarse edges
    edges_coarse = cv2.Canny(blurred2, int(0.6*median), int(1.6*median))
    
    # Sobel gradients
    sobelx = cv2.Sobel(roi_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = np.uint8(255 * sobel_mag / (np.max(sobel_mag) + 1e-5))
    _, sobel_edges = cv2.threshold(sobel_mag, int(0.3*median), 255, cv2.THRESH_BINARY)
    
    # Laplacian
    laplacian = cv2.Laplacian(roi_img, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.absolute(laplacian))
    _, lap_edges = cv2.threshold(laplacian, int(0.2*median), 255, cv2.THRESH_BINARY)
    
    # Weighted combination
    edges_combined = cv2.bitwise_or(edges_fine, edges_medium)
    edges_combined = cv2.bitwise_or(edges_combined, edges_coarse)
    edges_combined = cv2.bitwise_or(edges_combined, sobel_edges)
    edges_combined = cv2.bitwise_or(edges_combined, lap_edges)
    
    return edges_combined, edges_fine, edges_medium, sobel_mag

def crack_specific_morphology(edges):
    """Morphological operations specifically designed for cracks."""
    # Small circular noise removal
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_circle, iterations=1)
    
    # Detect vertical cracks
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel_v, iterations=1)
    
    # Detect horizontal cracks (less common but possible)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontal = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel_h, iterations=1)
    
    # Detect diagonal cracks
    kernel_d1 = np.array([[1,0,0,0,0],
                          [0,1,0,0,0],
                          [0,0,1,0,0],
                          [0,0,0,1,0],
                          [0,0,0,0,1]], dtype=np.uint8)
    diagonal1 = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel_d1, iterations=1)
    
    kernel_d2 = np.flip(kernel_d1, axis=1)
    diagonal2 = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel_d2, iterations=1)
    
    # Combine all directional cracks
    linear_cracks = cv2.bitwise_or(vertical, horizontal)
    linear_cracks = cv2.bitwise_or(linear_cracks, diagonal1)
    linear_cracks = cv2.bitwise_or(linear_cracks, diagonal2)
    
    # Fill small gaps within cracks
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    closed = cv2.morphologyEx(linear_cracks, cv2.MORPH_CLOSE, kernel_line, iterations=2)
    
    # Skeleton gives single-pixel-wide crack representation
    skeleton = skeletonize((closed > 0).astype(np.uint8))
    skeleton = (skeleton * 255).astype(np.uint8)
    
    # Remove very short line segments (likely noise, not cracks)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    
    cleaned_skeleton = np.zeros_like(skeleton)
    min_length = 20  # Minimum crack length in pixels
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_length:
            cleaned_skeleton[labels == i] = 255
    
    return opened, closed, linear_cracks, skeleton, cleaned_skeleton

def validate_and_localize_cracks(skeleton, original_img):
    """Smart crack detection with validation."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8
    )
    
    # Strict validation criteria
    min_length = 25  # Minimum pixels
    min_aspect_ratio = 3  # Cracks are elongated
    max_width = 8  # Cracks are thin
    
    valid_cracks = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        aspect_ratio = max(width, height) / (min(width, height) + 1)
        max_dim = max(width, height)
        
        # Validation checks
        is_long_enough = area >= min_length
        is_elongated = aspect_ratio >= min_aspect_ratio
        is_thin_enough = min(width, height) <= max_width
        is_significant = max_dim >= 30
        
        if is_long_enough and is_elongated and is_thin_enough and is_significant:
            # Calculate crack orientation
            crack_mask = (labels == i).astype(np.uint8) * 255
            crack_pixels = np.column_stack(np.where(crack_mask > 0))
            
            # Fit line to determine crack orientation
            if len(crack_pixels) > 10:
                vx, vy, x0, y0 = cv2.fitLine(crack_pixels.astype(np.float32), 
                                             cv2.DIST_L2, 0, 0.01, 0.01)
                angle = np.arctan2(vy[0], vx[0]) * 180 / np.pi
            else:
                angle = 0
            
            valid_cracks.append({
                'label': i,
                'length': area,
                'width': min(width, height),
                'height': max(width, height),
                'centroid': centroids[i],
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                'aspect_ratio': aspect_ratio,
                'orientation_deg': angle
            })
    
    # Create visualization (annotated image)
    result = original_img.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    crack_overlay = np.zeros_like(result)
    total_length = 0
    max_length = 0
    
    for idx, crack in enumerate(valid_cracks, 1):
        # Get crack pixels
        crack_mask = (labels == crack['label']).astype(np.uint8) * 255
        crack_pixels = np.column_stack(np.where(crack_mask > 0))
        
        total_length += len(crack_pixels)
        max_length = max(max_length, len(crack_pixels))
        
        # Draw crack in bright red
        for y, x in crack_pixels:
            cv2.circle(crack_overlay, (x, y), 2, (255, 0, 0), -1)
        
        # Draw bounding box
        x, y, w, h = crack['bbox']
        cv2.rectangle(result, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 3)
        
        # Label
        label_text = f"Crack #{idx}"
        cv2.putText(result, label_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw centroid
        cx, cy = int(crack['centroid'][0]), int(crack['centroid'][1])
        cv2.circle(result, (cx, cy), 5, (255, 255, 0), -1)
        
        # Draw orientation line
        angle_rad = crack['orientation_deg'] * np.pi / 180
        line_length = 40
        x2 = int(cx + line_length * np.cos(angle_rad))
        y2 = int(cy + line_length * np.sin(angle_rad))
        cv2.line(result, (cx, cy), (x2, y2), (255, 0, 255), 2)
    
    # Blend overlay
    result = cv2.addWeighted(result, 0.6, crack_overlay, 0.4, 0)
    
    # Calculate risk
    if total_length > 200:
        risk = 'SEVERE'
    elif total_length > 100:
        risk = 'HIGH'
    elif total_length > 50:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'
    
    metrics = {
        'num_cracks': len(valid_cracks),
        'total_length_px': total_length,
        'longest_crack_px': max_length,
        'average_length_px': total_length // len(valid_cracks) if len(valid_cracks) > 0 else 0,
        'crack_details': valid_cracks,
        'clinical_significance': len(valid_cracks) > 0 and total_length > 50,
        'risk_level': risk
    }
    
    return result, metrics, valid_cracks

def process_image_complete(img, filename):
    """Complete processing pipeline - returns annotated image and metrics only."""
    try:
        # [1] Preprocessing
        gray, enhanced, denoised, sharpened = advanced_preprocessing(img)
        
        # [2] Teeth Region Segmentation (try simple first, fallback to others)
        roi, mask = segment_teeth_simple_geometric(sharpened)
        
        if np.sum(roi > 0) < 1000:
            roi, mask = segment_teeth_adaptive(sharpened)
        
        if np.sum(roi > 0) < 1000:
            roi, mask = segment_teeth_region_balanced(sharpened)
        
        # [3] Edge Detection
        edges, edges_fine, edges_med, sobel = detect_cracks_advanced(roi)
        
        # [4] Morphological Processing
        opened, closed, linear, skeleton, clean_skel = crack_specific_morphology(edges)
        
        # [5] Crack Validation & Annotation
        result, metrics, cracks = validate_and_localize_cracks(clean_skel, img)
        
        return result, metrics  # No viz/save here - handled in UI
    
    except Exception as e:
        raise ValueError(f"Processing failed: {str(e)}")