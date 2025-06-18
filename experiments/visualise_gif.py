import os
import argparse
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def extract_frame_difference(frame1, frame2, alpha=0.8, sigma=5):
    """Extract the difference between two frames and return it with transparency."""
    # Convert frames to RGBA if they aren't already
    if frame1.mode != 'RGBA':
        frame1 = frame1.convert('RGBA')
    if frame2.mode != 'RGBA':
        frame2 = frame2.convert('RGBA')
    
    # Calculate difference between frames
    diff = ImageChops.difference(frame1, frame2)
    
    # Create mask where pixels have changed
    threshold = 50  # Adjust this value to control sensitivity
    diff_array = np.array(diff)
    mask = np.any(diff_array[:, :, :3] > threshold, axis=2)
    
    # Create transparent image for result
    result = Image.new('RGBA', frame1.size, (0, 0, 0, 0))
    result_array = np.array(result)
    frame2_array = np.array(frame2)
    
    if mask.any():  # Only process if there are changed pixels
        # Create color array from frame2
        color_array = frame2_array[:, :, :3].astype(float)
        
        # Apply Gaussian blur to each color channel
        blurred_colors = np.stack([
            gaussian_filter(color_array[:, :, i], sigma=sigma)
            for i in range(3)
        ], axis=2)
        
        # Add some random noise to make it more natural
        noise = np.random.normal(0, 5, blurred_colors.shape)
        blurred_colors = np.clip(blurred_colors + noise, 0, 255)
        
        # Convert back to uint8
        blurred_colors = blurred_colors.astype(np.uint8)
        
        # Create RGBA array with the specified alpha
        rgba_colors = np.zeros(frame2_array.shape, dtype=np.uint8)
        rgba_colors[..., :3] = blurred_colors
        rgba_colors[..., 3] = int(255 * alpha)
        
        # Apply the blurred colors only to the masked area
        result_array[mask] = rgba_colors[mask]
    
    return Image.fromarray(result_array)

def create_motion_trail(gif_path, output_path=None, alpha=0.5, sigma=10):
    """Create a composite image showing motion trail from a GIF."""
    # Open GIF file
    gif = Image.open(gif_path)
    n_frames = gif.n_frames
    
    # Get first frame as base with full opacity
    gif.seek(0)
    composite = gif.copy().convert('RGBA')
    # Ensure first frame is fully opaque
    composite.putalpha(255)  # Set alpha to 255 (fully opaque)
    
    # Process each subsequent frame
    for i in tqdm(range(1, n_frames), desc="Processing frames", unit="frame"):
        gif.seek(i-1)
        frame1 = gif.copy()
        gif.seek(i)
        frame2 = gif.copy()
        
        # Extract difference and blend it into composite
        diff = extract_frame_difference(frame1, frame2, alpha=alpha, sigma=sigma)
        composite = Image.alpha_composite(composite, diff)
    
    # If no output path specified, create one based on input path
    if output_path is None:
        base_path = os.path.splitext(gif_path)[0]
        output_path = f"{base_path}_motion_trail.png"
    
    # Save result
    composite.save(output_path, 'PNG')
    print(f"Motion trail image saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Create motion trail image from GIF')
    parser.add_argument('gif_path', help='Path to input GIF file')
    parser.add_argument('--output', '-o', help='Path for output PNG file (optional)')
    parser.add_argument('--alpha', '-a', type=float, default=0.8,
                      help='Alpha value for fade effect (default: 0.5)')
    parser.add_argument('--sigma', '-s', type=float, default=5,
                      help='Sigma value for Gaussian blur (default: 10)')
    
    args = parser.parse_args()
    
    create_motion_trail(args.gif_path, args.output, args.alpha, args.sigma)

if __name__ == "__main__":
    main()