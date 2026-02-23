from PIL import Image
import struct
import glob
import os
import numpy as np
from sklearn.decomposition import PCA
# --- Argument parsing and main entry point ---
import argparse
import sys
from sklearn.cluster import KMeans

# Optional: prefer cuML KMeans if available (GPU acceleration). Fallback to sklearn KMeans.
try:
    from cuml.cluster import KMeans as cuKMeans
    _USE_CUML = True
except Exception:
    cuKMeans = None
    _USE_CUML = False

def make_kmeans(n_clusters, **kwargs):
    """Return a KMeans instance. Uses cuML if available, otherwise sklearn's KMeans.

    kwargs may include random_state and n_init; cuML may ignore n_init.
    """
    if _USE_CUML:
        # cuML KMeans accepts n_clusters and random_state; other kwargs are ignored.
        return cuKMeans(n_clusters=n_clusters, random_state=kwargs.get('random_state', 42))
    else:
        return KMeans(n_clusters=n_clusters, random_state=kwargs.get('random_state', 42), n_init=kwargs.get('n_init', 10))

# Stub for font_to_edge_descriptor
def font_to_edge_descriptor(font_bytes):
    # Simple edge descriptor: count horizontal and vertical transitions in the 8x8 pattern
    arr = np.array(font_bytes, dtype=np.uint8)
    # Convert to 8x8 binary matrix
    mat = np.zeros((8,8), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            mat[row, col] = (arr[row] >> (7-col)) & 1
    # Horizontal transitions
    horiz = np.sum(mat[:, :-1] != mat[:, 1:])
    # Vertical transitions
    vert = np.sum(mat[:-1, :] != mat[1:, :])
    # Flattened edge map (difference between neighbors)
    edge_map = np.zeros(64, dtype=np.float32)
    idx = 0
    for row in range(8):
        for col in range(8):
            val = 0
            if col < 7:
                val += mat[row, col] != mat[row, col+1]
            if row < 7:
                val += mat[row, col] != mat[row+1, col]
            edge_map[idx] = val
            idx += 1
    # Add transitions as first two features
    edge_map[0] = horiz
    edge_map[1] = vert
    return edge_map

# Stub for font_density_features
def font_density_features(font_bytes):
    # Simple density features: total set bits, per-row density, per-col density
    arr = np.array(font_bytes, dtype=np.uint8)
    mat = np.zeros((8,8), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            mat[row, col] = (arr[row] >> (7-col)) & 1
    total_on = np.sum(mat)
    row_density = np.sum(mat, axis=1) / 8.0
    col_density = np.sum(mat, axis=0) / 8.0
    # Compose feature vector: [total_on, row_density (8), col_density (8)]
    feats = np.zeros(17, dtype=np.float32)
    feats[0] = total_on / 64.0
    feats[1:9] = row_density
    feats[9:17] = col_density
    return feats

# Stub for compute_medoids
def compute_medoids(fonts, labels, n_clusters):
    # Compute medoid for each cluster: the member with minimum total Hamming distance
    # fonts: array shape (n_fonts, 8) where each row is 8 bytes representing 8x8 bits
    medoids = []
    n_fonts = fonts.shape[0]
    # Precompute bit matrices for fast Hamming distance: (n_fonts, 64) boolean
    bitmats = np.zeros((n_fonts, 64), dtype=np.uint8)
    for i in range(n_fonts):
        arr = fonts[i]
        bits = []
        for r in range(8):
            byte = int(arr[r])
            for c in range(8):
                bits.append((byte >> (7 - c)) & 1)
        bitmats[i] = np.array(bits, dtype=np.uint8)

    for k in range(n_clusters):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            medoids.append(np.zeros(8, dtype=np.uint8))
            continue
        # Extract cluster bitmats
        cluster_bits = bitmats[idxs]  # (m,64)
        # Compute pairwise Hamming distances: using matrix multiplication
        # distance between a and b = sum(a != b) = sum(a) + sum(b) - 2*sum(a & b)
        sums = cluster_bits.sum(axis=1).astype(np.int32)
        # Compute overlap matrix
        overlap = cluster_bits.dot(cluster_bits.T)  # (m,m)
        # pairwise distances
        m = cluster_bits.shape[0]
        dists = np.zeros((m, m), dtype=np.int32)
        for i in range(m):
            # d(i,j) = sums[i] + sums[j] - 2*overlap[i,j]
            dists[i, :] = sums[i] + sums - 2 * overlap[i, :]
        # total distance for each member
        total = dists.sum(axis=1)
        medoid_idx = int(np.argmin(total))
        medoids.append(fonts[idxs[medoid_idx]])

    medoids = np.array(medoids)

    # Ensure the medoid set always contains a completely-empty (all-0) and
    # a completely-filled (all-1 / 0xFF) pattern. Some frames (or palettes)
    # may be entirely black or white and we need explicit font entries so the
    # renderer can pick an exact all-0x00 or all-0xFF character.
    # Strategy: if no medoid equals all-0x00, replace the medoid with the
    # smallest population (fewest set bits) with an all-0x00 pattern. If no
    # medoid equals all-0xFF, replace the medoid with the largest population
    # (most set bits) with an all-0xFF pattern. Try to pick distinct indices
    # when both replacements are required.
    try:
        # Compute ones count per medoid
        ones_counts = np.zeros(len(medoids), dtype=np.int32)
        for i in range(len(medoids)):
            bits = 0
            for b in medoids[i]:
                bits += bin(int(b)).count('1')
            ones_counts[i] = bits

        # Flags whether exact patterns already present
        has_all_zero = np.any(ones_counts == 0)
        has_all_one = np.any(ones_counts == 64)

        replace_zero_idx = None
        replace_one_idx = None

        if not has_all_zero:
            # pick medoid with minimal ones (fewest set bits)
            replace_zero_idx = int(np.argmin(ones_counts))
            medoids[replace_zero_idx] = np.zeros(8, dtype=np.uint8)
            ones_counts[replace_zero_idx] = 0

        if not has_all_one:
            # If only one medoid exists and it already represents all-zero, keep it.
            if len(medoids) == 1 and ones_counts[0] == 0:
                # Not enough capacity to also store all-ones; skip enforcement.
                pass
            else:
                # pick medoid with maximal ones (most set bits)
                # avoid picking the same index chosen for zero replacement when possible
                if replace_zero_idx is None:
                    replace_one_idx = int(np.argmax(ones_counts))
                else:
                    if len(medoids) == 1:
                        replace_one_idx = replace_zero_idx
                    else:
                        argmax = int(np.argmax(ones_counts))
                        if argmax != replace_zero_idx:
                            replace_one_idx = argmax
                        else:
                            tmp = ones_counts[argmax]
                            ones_counts[argmax] = -1
                            replace_one_idx = int(np.argmax(ones_counts))
                            ones_counts[argmax] = tmp
                if replace_one_idx is not None:
                    medoids[replace_one_idx] = np.array([0xFF] * 8, dtype=np.uint8)

    except Exception:
        # Be conservative: if anything unexpected happens just return medoids as-is
        return medoids

    return medoids

def main():
    parser = argparse.ArgumentParser(description="K2 MEMTEXT animation converter")
    parser.add_argument('--animation', type=str, required=True, help='Input directory for animation frames')
    parser.add_argument('--output-bin', type=str, required=True, help='Output binary file (animation mode)')
    parser.add_argument('--frame-duration', type=int, default=6, help='Frame duration in 60Hz ticks')
    parser.add_argument('--rle', action='store_true', help='Enable RLE encoding for binary output (where supported)')
    args = parser.parse_args()

    print("K2 Memtext Image Converter")
    print("============================")
    print("Animation mode")
    print(f"Processing directory: {args.animation}")
    print("MEMTEXT mode enabled: 256-color palettes, 1024 medoids in 4 font sets, pattern inversion")

    frames_data = process_animation_frames_memtext(args.animation)
    if not frames_data:
        print("Error: No frames processed successfully")
        sys.exit(1)

    print("Saving memtext animation output...")
    save_output_bin_memtext(frames_data, args.output_bin, frame_duration=args.frame_duration, use_rle=args.rle)
    print("MEMTEXT animation conversion completed successfully!")


def load_and_scale_image(input_path):
    """Load and scale image to 640x480 with aspect ratio preservation."""
    img = Image.open(input_path).convert('RGB')
    target_size = (640, 480)
    
    # Calculate new size preserving aspect ratio
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)
    
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create black background and center the image
    background = Image.new('RGB', target_size, (0, 0, 0))
    offset_x = (target_size[0] - new_width) // 2
    offset_y = (target_size[1] - new_height) // 2
    background.paste(img_resized, (offset_x, offset_y))
    
    return background

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space for perceptual distance calculation."""
    # Proper RGB to LAB conversion
    rgb = rgb.astype(np.float32) / 255.0
    
    # Apply gamma correction
    mask = rgb > 0.04045
    rgb = np.where(mask, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    
    # Convert to XYZ using sRGB matrix
    xyz = np.zeros_like(rgb)
    xyz[..., 0] = 0.4124564 * rgb[..., 0] + 0.3575761 * rgb[..., 1] + 0.1804375 * rgb[..., 2]
    xyz[..., 1] = 0.2126729 * rgb[..., 0] + 0.7151522 * rgb[..., 1] + 0.0721750 * rgb[..., 2]
    xyz[..., 2] = 0.0193339 * rgb[..., 0] + 0.1191920 * rgb[..., 1] + 0.9503041 * rgb[..., 2]
    
    # Normalize by D65 illuminant
    xyz[..., 0] /= 0.95047
    xyz[..., 1] /= 1.00000
    xyz[..., 2] /= 1.08883
    
    # Convert to LAB
    mask = xyz > 0.008856
    xyz = np.where(mask, np.power(xyz, 1/3), (7.787 * xyz + 16/116))
    
    lab = np.zeros_like(xyz)
    lab[..., 0] = 116 * xyz[..., 1] - 16  # L
    lab[..., 1] = 500 * (xyz[..., 0] - xyz[..., 1])  # a
    lab[..., 2] = 200 * (xyz[..., 1] - xyz[..., 2])  # b
    
    return lab


def invert_pattern(pattern):
    """Invert an 8-byte pattern (XOR each byte with 0xFF)."""
    return np.array([b ^ 0xFF for b in pattern], dtype=np.uint8)


def pattern_key(pattern):
    """Return a hashable key for a pattern."""
    return tuple(pattern.tolist())


def get_canonical_pattern(pattern):
    """Return the canonical form of a pattern (lower of pattern or its inverse).
    Also returns whether the canonical is inverted (True if inverse was used)."""
    p = np.array(pattern, dtype=np.uint8)
    inv = invert_pattern(p)
    # Use lexicographic comparison
    if tuple(inv.tolist()) < tuple(p.tolist()):
        return inv, True
    return p, False


def create_memtext_256_color_palette(image_files):
    """Create a 256-color palette from all animation frames using KMeans.
    Palette index 0 is reserved for transparency (set to black).
    FG and BG palettes are identical.
    
    Returns: palette as np.array of shape (256, 3), dtype=uint8
    """
    print("Creating 256-color palette for MEMTEXT mode...")
    
    # Sample pixels from all frames
    all_pixels = []
    sample_rate = 4  # Sample every 4th pixel to speed up
    
    for idx, image_file in enumerate(image_files):
        img = Image.open(image_file).convert('RGB')
        img_np = np.array(img)
        pixels = img_np.reshape(-1, 3)
        # Subsample for speed
        all_pixels.append(pixels[::sample_rate])
    
    all_pixels = np.vstack(all_pixels).astype(np.float32)
    print(f"  Sampled {len(all_pixels)} pixels from {len(image_files)} frames")
    
    # Remove duplicates
    unique_pixels = np.unique(all_pixels, axis=0)
    print(f"  Found {len(unique_pixels)} unique colors")
    
    # Cluster to 255 colors (leaving index 0 for transparency)
    n_clusters = min(255, len(unique_pixels))
    if n_clusters < 255:
        # If fewer unique colors, use them directly
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[1:n_clusters+1] = unique_pixels[:n_clusters].astype(np.uint8)
        # Pad with grey
        for i in range(n_clusters+1, 256):
            palette[i] = [128, 128, 128]
    else:
        kmeans = make_kmeans(n_clusters=255, random_state=42, n_init=10)
        kmeans.fit(unique_pixels)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        
        palette = np.zeros((256, 3), dtype=np.uint8)
        # Index 0 reserved for transparency (black)
        palette[1:256] = centers
    
    print(f"  Created 256-color palette (index 0 = transparent/black)")
    return palette


def process_animation_frames_memtext(input_dir):
    """Process animation frames in MEMTEXT mode.
    
    Key differences from regular animation mode:
    - 256-color FG/BG palettes (identical)
    - 1024 medoids in 4 font sets of 256 each
    - Pattern inversion support to reduce unique patterns
    - Tile data includes font_set (0-3), char_index (0-255), and invert bit
    
    Args:
        input_dir: Directory containing input frames
    """
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"Error: No image files found in directory {input_dir}")
        return []
    
    print(f"Found {len(image_files)} image files")
    
    # Step 1: Create 256-color global palette
    palette = create_memtext_256_color_palette(image_files)
    
    # Step 2: Load all frames and extract unique 8x8 patterns
    print("Extracting 8x8 patterns from all frames...")
    all_tiles = []
    frame_tile_indices = []  # (start, end) for each frame
    
    
    for idx, image_file in enumerate(image_files):
        print(f"Loading frame {idx+1}/{len(image_files)}: {os.path.basename(image_file)}")
        image = load_and_scale_image(image_file)
        
        img_np = np.array(image)
        block_height = 24
        num_blocks = 20
        chars_per_block_row = 3
        chars_per_block = chars_per_block_row * 80
        total_chars = num_blocks * chars_per_block
        
        frame_tiles = []
        for i in range(total_chars):
            block_idx = i // chars_per_block
            char_in_block = i % chars_per_block
            char_row = char_in_block // 80
            char_col = char_in_block % 80
            y_start = block_idx * block_height
            y0 = y_start + char_row * 8
            x0 = char_col * 8
            char_img = img_np[y0:y0+8, x0:x0+8, :]
            
            # Generate initial font pattern using local optimization
            # We use the 256-color palette
            font_def, _, _ = optimize_character_memtext(char_img, palette)
            
            frame_tiles.append({
                'font': np.array(font_def, dtype=np.uint8),
                'pixels': np.ascontiguousarray(char_img.astype(np.uint8))
            })
        
        start_idx = len(all_tiles)
        all_tiles.extend(frame_tiles)
        end_idx = len(all_tiles)
        frame_tile_indices.append((start_idx, end_idx))
    
    print(f"Extracted {len(all_tiles)} total tiles")
    
    # Step 3: Extract unique patterns considering inversion
    print("Finding unique patterns (considering inversions)...")
    unique_canonical = {}  # canonical_key -> (canonical_pattern, original_was_inverted)
    
    for tile in all_tiles:
        pattern = tile['font']
        canonical, was_inverted = get_canonical_pattern(pattern)
        key = pattern_key(canonical)
        if key not in unique_canonical:
            unique_canonical[key] = canonical
    
    print(f"Found {len(unique_canonical)} unique canonical patterns (after inversion reduction)")
    
    # Step 4: Cluster to 1024 medoids
    print("Clustering to 1024 medoids...")
    patterns_list = list(unique_canonical.values())
    
    # Always include all-zeros pattern
    zero_tile = np.zeros(8, dtype=np.uint8)
    zero_key = pattern_key(zero_tile)
    if zero_key not in unique_canonical:
        patterns_list.insert(0, zero_tile)
    
    patterns_arr = np.array(patterns_list, dtype=np.uint8)
    
    # Target 1024 medoids
    n_medoids = 1024
    cluster_target = min(len(patterns_arr), n_medoids)
    
    if cluster_target < n_medoids:
        # If fewer patterns than 1024, use them all and pad
        clustered_medoids = patterns_arr.copy()
        if len(clustered_medoids) < n_medoids:
            # Pad with zeros
            pad_count = n_medoids - len(clustered_medoids)
            pad = np.zeros((pad_count, 8), dtype=np.uint8)
            clustered_medoids = np.vstack([clustered_medoids, pad])
    else:
        # Cluster using features
        n_unique = patterns_arr.shape[0]
        edge_feats = np.zeros((n_unique, 64), dtype=np.float32)
        density_feats = np.zeros((n_unique, 17), dtype=np.float32)
        raw_bytes = np.zeros((n_unique, 8), dtype=np.float32)
        
        for i in range(n_unique):
            edge_feats[i] = font_to_edge_descriptor(patterns_arr[i])
            density_feats[i] = font_density_features(patterns_arr[i])
            raw_bytes[i] = patterns_arr[i].astype(np.float32)
        
        def norm_feats(feats):
            mean = feats.mean(axis=0)
            std = feats.std(axis=0) + 1e-6
            return (feats - mean) / std
        
        edge_feats_n = norm_feats(edge_feats) * 0.1
        density_feats_n = norm_feats(density_feats) * 0.1
        raw_bytes_n = norm_feats(raw_bytes) * 10.0
        
        perceptual_descriptors = np.concatenate([edge_feats_n, density_feats_n, raw_bytes_n], axis=1)
        
        # PCA for dimensionality reduction
        try:
            pca = PCA(n_components=min(perceptual_descriptors.shape))
            pca.fit(perceptual_descriptors)
            font_features = pca.transform(perceptual_descriptors)
        except Exception:
            font_features = perceptual_descriptors
        
        # Cluster
        kmeans = make_kmeans(n_clusters=cluster_target, random_state=42, n_init=10)
        kmeans.fit(font_features)
        cluster_labels = kmeans.labels_
        
        # Compute medoids
        clustered_medoids = compute_medoids(patterns_arr, cluster_labels, cluster_target)
    
    print(f"Created {len(clustered_medoids)} medoids")
    
    # Ensure we have exactly 1024 medoids
    if len(clustered_medoids) < 1024:
        pad_count = 1024 - len(clustered_medoids)
        pad = np.zeros((pad_count, 8), dtype=np.uint8)
        clustered_medoids = np.vstack([clustered_medoids, pad])
    elif len(clustered_medoids) > 1024:
        clustered_medoids = clustered_medoids[:1024]
    
    # Split into 4 font sets of 256 each
    font_sets = [clustered_medoids[i*256:(i+1)*256] for i in range(4)]
    
    print(f"Split medoids into 4 font sets of 256 patterns each")
    
    # Precompute medoid masks for error-based matching
    all_medoid_masks = np.zeros((1024, 64), dtype=bool)
    for m_idx in range(1024):
        pattern = clustered_medoids[m_idx]
        bits = []
        for r in range(8):
            byte = int(pattern[r])
            for c in range(8):
                bits.append((byte >> (7 - c)) & 1)
        all_medoid_masks[m_idx] = np.array(bits, dtype=bool)
    
    edge_pixel_weights = np.ones(64, dtype=np.float32)
    
    # Step 6: Process each frame - assign tiles to medoids + colors
    print("Assigning tiles to medoids for each frame...")
    frames_data = []
    
    # Precompute medoid bit patterns for fast Hamming distance
    medoid_bits = np.zeros((1024, 64), dtype=np.uint8)
    for m_idx in range(1024):
        pattern = clustered_medoids[m_idx]
        for r in range(8):
            byte = int(pattern[r])
            for c in range(8):
                medoid_bits[m_idx, r * 8 + c] = (byte >> (7 - c)) & 1
    
    # Precompute palette as float for fast matching
    palette_f = palette.astype(np.float32)
    
    for frame_idx, (start, end) in enumerate(frame_tile_indices):
        print(f"Processing frame {frame_idx+1}/{len(frame_tile_indices)}...")
        frame_tiles = all_tiles[start:end]
        n_tiles = len(frame_tiles)
        
        # For MEMTEXT, we need to output:
        # - tile_data: array of 16-bit values encoding (font_set:2, char_index:8, fg:8, bg:8, invert:1) 
        # Store (font_set, char_index, fg_idx, bg_idx, invert_bit) per tile
        
        tile_assignments = np.zeros((n_tiles, 5), dtype=np.uint16)
        
        # For each tile, find best medoid (considering inversion) and best FG/BG
        for i, tile in enumerate(frame_tiles):
            orig_rgb = tile['pixels'].reshape(-1, 3).astype(np.float32)
            
            # First, determine FG and BG colors from the tile pixels using simpler clustering
            unique_colors = np.unique(orig_rgb, axis=0)
            if len(unique_colors) < 2:
                # Uniform tile - find closest palette color
                avg = orig_rgb.mean(axis=0)
                dists = np.linalg.norm(palette_f - avg, axis=1)
                dists[0] = float('inf')  # Skip transparent
                best_color = int(np.argmin(dists))
                # Use all-zeros medoid with this color as both fg and bg
                tile_assignments[i] = [0, 0, best_color, best_color, 0]
                continue
            
            # Use simple threshold-based segmentation (faster than k-means)
            luminance = 0.299 * orig_rgb[:, 0] + 0.587 * orig_rgb[:, 1] + 0.114 * orig_rgb[:, 2]
            threshold = np.median(luminance)
            
            fg_mask = luminance >= threshold
            bg_mask = ~fg_mask
            
            # Compute average colors for each region
            if np.any(fg_mask):
                fg_center = orig_rgb[fg_mask].mean(axis=0)
            else:
                fg_center = orig_rgb.mean(axis=0)
            if np.any(bg_mask):
                bg_center = orig_rgb[bg_mask].mean(axis=0)
            else:
                bg_center = orig_rgb.mean(axis=0)
            
            # Map centers to palette
            fg_dists = np.linalg.norm(palette_f - fg_center, axis=1)
            fg_dists[0] = float('inf')
            fg_idx = int(np.argmin(fg_dists))
            
            bg_dists = np.linalg.norm(palette_f - bg_center, axis=1)
            bg_idx = int(np.argmin(bg_dists))
            
            if fg_idx == bg_idx:
                bg_dists[bg_idx] = float('inf')
                bg_idx = int(np.argmin(bg_dists))
            
            # Build ideal font pattern from mask
            ideal_pattern = np.zeros(8, dtype=np.uint8)
            for row in range(8):
                byte = 0
                for col in range(8):
                    pixel_idx = row * 8 + col
                    if fg_mask[pixel_idx]:
                        byte |= (1 << (7 - col))
                ideal_pattern[row] = byte
            
            # Convert ideal pattern to bits
            ideal_bits = np.zeros(64, dtype=np.uint8)
            for r in range(8):
                byte = ideal_pattern[r]
                for c in range(8):
                    ideal_bits[r * 8 + c] = (byte >> (7 - c)) & 1

            # Find closest medoid using weighted Hamming distance
            # XOR gives 1 where bits differ
            xor_normal = np.bitwise_xor(medoid_bits, ideal_bits)
            # Weighted Hamming: multiply by edge weights before summing
            hamming_normal = np.sum(xor_normal * edge_pixel_weights, axis=1)
            
            # Also check inverted
            ideal_bits_inv = 1 - ideal_bits
            xor_inv = np.bitwise_xor(medoid_bits, ideal_bits_inv)
            hamming_inv = np.sum(xor_inv * edge_pixel_weights, axis=1)
            
            # Find best
            best_normal_idx = int(np.argmin(hamming_normal))
            best_inv_idx = int(np.argmin(hamming_inv))
            
            if hamming_normal[best_normal_idx] <= hamming_inv[best_inv_idx]:
                best_medoid_idx = best_normal_idx
                best_invert = 0
            else:
                best_medoid_idx = best_inv_idx
                best_invert = 1
            
            fs = best_medoid_idx // 256
            char_idx = best_medoid_idx % 256
            
            # Store assignment - fg/bg stay the same regardless of inversion
            # The invert bit tells the renderer to flip pattern bits, which
            # effectively swaps which pixels get fg vs bg colors
            tile_assignments[i] = [fs, char_idx, fg_idx, bg_idx, best_invert]
        
        frames_data.append({
            'width': 640,
            'height': 480,
            'use_single_row_blocks': False,
            'memtext': True,
            'tile_assignments': tile_assignments,  # (n_tiles, 5): fs, char, fg, bg, invert
            'palette': palette,
            'font_sets': font_sets,
        })
    
    # Store shared data in first frame for output
    frames_data[0]['global_palette'] = palette
    frames_data[0]['global_font_sets'] = font_sets
    
    return frames_data


def optimize_character_memtext(char_img, palette):
    """Optimize a character tile using 256-color palette.
    Returns (font_def, fg_idx, bg_idx) where font_def is 8 bytes.
    
    Args:
        char_img: 8x8x3 RGB pixel array
        palette: 256x3 color palette
    """
    # Flatten to 64 pixels
    pixels = char_img.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)
    
    # Find two most distinct colors in the tile using luminance thresholding
    unique_colors = np.unique(pixels, axis=0)
    if len(unique_colors) < 2:
        # Uniform tile
        avg_color = pixels.mean(axis=0)
        # Find closest palette color
        dists = np.linalg.norm(palette_f - avg_color, axis=1)
        dists[0] = float('inf')  # Skip transparent
        best_idx = int(np.argmin(dists))
        return np.zeros(8, dtype=np.uint8), best_idx, 0
    
    # Use luminance-based thresholding
    luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    
    threshold = np.median(luminance)
    
    fg_mask = luminance >= threshold
    bg_mask = ~fg_mask
    
    # Compute average colors for each region
    if np.any(fg_mask):
        fg_center = pixels[fg_mask].mean(axis=0)
    else:
        fg_center = pixels.mean(axis=0)
    if np.any(bg_mask):
        bg_center = pixels[bg_mask].mean(axis=0)
    else:
        bg_center = pixels.mean(axis=0)
    
    # Map centers to closest palette colors
    fg_dists = np.linalg.norm(palette_f - fg_center, axis=1)
    fg_dists[0] = float('inf')  # Skip transparent
    fg_idx = int(np.argmin(fg_dists))
    
    bg_dists = np.linalg.norm(palette_f - bg_center, axis=1)
    bg_idx = int(np.argmin(bg_dists))
    
    # Ensure fg != bg
    if fg_idx == bg_idx:
        bg_dists[bg_idx] = float('inf')
        bg_idx = int(np.argmin(bg_dists))
    
    # Build font pattern: 1 where fg_mask, 0 where bg_mask
    font_def = np.zeros(8, dtype=np.uint8)
    for row in range(8):
        byte = 0
        for col in range(8):
            pixel_idx = row * 8 + col
            if fg_mask[pixel_idx]:
                byte |= (1 << (7 - col))
        font_def[row] = byte
    
    return font_def, fg_idx, bg_idx

def save_output_bin_memtext(frames_data, output_bin, frame_duration=6, use_rle=False):
    """Save binary output in MEMTEXT format.
    
    Key differences from regular format:
    - Text Color LUT: 2048 bytes (256 colors * 4 bytes BGRA, FG followed by BG but they're identical)
    - Text Font Data: 4 chunks of 2048 bytes each (font sets 0-3)
    - Per frame: 5 character chunks, 5 color chunks (RLE if use_rle=True, fixed otherwise)
    """
    print(f"Saving MEMTEXT binary to {output_bin}...")
    
    # Get global data from first frame
    palette = frames_data[0]['global_palette']
    font_sets = frames_data[0]['global_font_sets']
    
    # Header
    magic = 0xA8
    version = 0x01
    frame_duration_byte = frame_duration & 0xFF
    mode = 0x02  # Memtext mode
    columns = 80
    rows = 60
    xoffset = 0
    yoffset = 0
    
    with open(output_bin, 'wb') as f:
        # Write header: 8 bytes
        f.write(struct.pack('BBBBBBBB', magic, version, frame_duration_byte, mode, columns, rows, xoffset, yoffset))
        
        # Write Text Color LUT for MEMTEXT (chunk type 0x01, 2048 bytes)
        # FG and BG are identical, each 256 colors * 4 bytes = 1024 bytes
        # Total = 2048 bytes
        palette_bgra = np.zeros((256, 4), dtype=np.uint8)
        palette_bgra[:, 0] = palette[:, 2]  # B
        palette_bgra[:, 1] = palette[:, 1]  # G
        palette_bgra[:, 2] = palette[:, 0]  # R
        palette_bgra[:, 3] = 255            # A
        palette_bgra[0, 3] = 0              # Transparent for index 0
        
        lut_data = palette_bgra.tobytes() + palette_bgra.tobytes()  # FG + BG (identical)
        f.write(struct.pack('BBH', 0x01, 0x00, len(lut_data)))
        f.write(lut_data)
        
        # Write 4 Font Data chunks (chunk type 0x02, chunk ID = font set id)
        for fs_id in range(4):
            font_data = font_sets[fs_id].astype(np.uint8).tobytes()
            if len(font_data) < 2048:
                font_data += b'\x00' * (2048 - len(font_data))
            f.write(struct.pack('BBH', 0x02, fs_id, len(font_data)))
            f.write(font_data)
        
        # Write frames
        for frame_idx, frame_data in enumerate(frames_data):
            print(f"Writing frame {frame_idx + 1}/{len(frames_data)}...")
            
            # Frame Start
            f.write(struct.pack('BBH', 0x00, 0x00, 0))
            
            tile_assignments = frame_data['tile_assignments']  # (4800, 5)
            
            # Build character data: 2 bytes per tile
            # Encoding: byte0 = char_index, byte1 = (font_set:2, invert:1, unused:5)
            # Actually, let's use: word = char_index | (font_set << 8) | (invert << 10)
            # For MEMTEXT per FileFormat.md, RLE encodes words (2 bytes) not single bytes
            
            char_data = []
            for i in range(len(tile_assignments)):
                fs = tile_assignments[i, 0]
                char_idx = tile_assignments[i, 1]
                invert = tile_assignments[i, 4]
                # Per spec: low byte [7:0] = CHARACTER, high byte [8:9] = FONT BANK, [12] = INVERT
                # Bit positions in high byte: [0:1] = font bank (bits 8-9), [4] = invert (bit 12)
                high_byte = (fs & 0x03) | ((invert & 0x01) << 4)
                word = (char_idx & 0xFF) | (high_byte << 8)
                char_data.append(word & 0xFF)
                char_data.append((word >> 8) & 0xFF)
            
            # Build color data: 2 bytes per tile
            # Per spec: low byte [7:0] = BACKGROUND, high byte [15:8] = FOREGROUND
            color_data = []
            for i in range(len(tile_assignments)):
                fg = tile_assignments[i, 2]
                bg = tile_assignments[i, 3]
                color_data.append(bg & 0xFF)   # Low byte = background
                color_data.append(fg & 0xFF)   # High byte = foreground
            
            char_data = bytes(char_data)
            color_data = bytes(color_data)
            
            # Split into 5 chunks: 4 of 2048 bytes + 1 of 1408 bytes
            # Total: 60 rows * 80 cols * 2 bytes = 9600 bytes
            chunk_sizes = [2048] * 4 + [1408]
            
            for chunk_idx in range(5):
                start = sum(chunk_sizes[:chunk_idx])
                end = start + chunk_sizes[chunk_idx]
                
                # Character chunk
                char_chunk = char_data[start:end] if end <= len(char_data) else char_data[start:] + b'\x00' * (end - len(char_data))
                chunk_id = chunk_idx
                if use_rle:
                    char_encoded = rle_encode_memtext(char_chunk)
                    f.write(struct.pack('BBH', 0x05, chunk_id, len(char_encoded)))
                    f.write(char_encoded)
                else:
                    f.write(struct.pack('BBH', 0x03, chunk_id, len(char_chunk)))
                    f.write(char_chunk)
                
                # Color chunk
                color_chunk = color_data[start:end] if end <= len(color_data) else color_data[start:] + b'\x00' * (end - len(color_data))
                if use_rle:
                    color_encoded = rle_encode_memtext(color_chunk)
                    f.write(struct.pack('BBH', 0x06, chunk_id, len(color_encoded)))
                    f.write(color_encoded)
                else:
                    f.write(struct.pack('BBH', 0x04, chunk_id, len(color_chunk)))
                    f.write(color_chunk)
            
            # Frame End
            f.write(struct.pack('BBH', 0xFF, 0x00, 0))
    
    print(f"MEMTEXT binary saved to {output_bin}")


def rle_encode_memtext(data):
    """RLE encode data for MEMTEXT mode (word-based).
    For MEMTEXT, each unit is 2 bytes (word).
    If bit 15 of count is clear: repeat next word count times.
    If bit 15 is set: output next (count & 0x7FFF) words without repetition.
    """
    result = bytearray()
    i = 0
    n = len(data) // 2  # Number of words
    
    def get_word(idx):
        return (data[idx*2], data[idx*2+1])
    
    while i < n:
        # Check for a run of the same word
        run_start = i
        run_word = get_word(i)
        j = i + 1
        while j < n and get_word(j) == run_word:
            j += 1
        run_length = j - run_start
        
        if run_length >= 2:
            # Emit run
            while run_length > 0:
                emit = min(run_length, 0x7FFF)
                result.extend(struct.pack('<H', emit))
                result.append(run_word[0])
                result.append(run_word[1])
                run_length -= emit
            i = j
        else:
            # Collect literals until next run >= 2 or end
            start = i
            i += 1
            while i < n:
                if i + 1 < n and get_word(i) == get_word(i + 1):
                    break
                i += 1
            literal_count = i - start
            
            # Emit literals in chunks of max 0x7FFF
            pos = start
            while literal_count > 0:
                emit = min(literal_count, 0x7FFF)
                result.extend(struct.pack('<H', 0x8000 | emit))
                for k in range(emit):
                    w = get_word(pos + k)
                    result.append(w[0])
                    result.append(w[1])
                pos += emit
                literal_count -= emit
    
    return bytes(result)


# Add main entry point
if __name__ == "__main__":
    main()

