#!/usr/bin/env python3
"""
Module to parse and render MEMTEXT binary animation files.

MEMTEXT format uses:
- 256-color palettes (FG and BG are identical)
- 4 font sets of 256 patterns each (1024 total)
- Per-tile: font_set, char_index, fg_color, bg_color, invert_bit
- RLE word-based encoding for character and color data

Usage:
    python render_memtext.py input.bin output_dir
"""
import os
import struct
import numpy as np
from PIL import Image
import argparse


def parse_header(f):
    """Parse 8-byte header."""
    data = f.read(8)
    if len(data) != 8:
        raise ValueError("File too short for header (expected 8 bytes)")
    magic, version, frame_duration, mode, columns, rows, x_offset, y_offset = struct.unpack('<BBBBBBBB', data)
    if magic != 0xA8:
        raise ValueError(f"Unexpected magic number: {magic:02X} (expected 0xA8)")
    return {
        'version': version,
        'frame_duration': frame_duration,
        'mode': mode,
        'columns': columns,
        'rows': rows,
        'x_offset': x_offset,
        'y_offset': y_offset
    }


def parse_chunk_header(f):
    """Parse 4-byte chunk header. Returns (type, id, length) or None at EOF."""
    data = f.read(4)
    if len(data) == 0:
        return None
    if len(data) != 4:
        raise ValueError(f"Unexpected end of file while reading chunk header (got {len(data)} bytes)")
    chunk_type, chunk_id, chunk_length = struct.unpack('<BBH', data)
    return chunk_type, chunk_id, chunk_length


def rle_decode_memtext(data, output_words):
    """Decode RLE data for MEMTEXT mode (word-based).
    
    Format: 2-byte count followed by word(s)
    - If bit 15 clear: repeat next word 'count' times
    - If bit 15 set: output next (count & 0x7FFF) words literally
    
    Returns: list of (low_byte, high_byte) tuples
    """
    result = []
    i = 0
    n = len(data)
    
    while i < n and len(result) < output_words:
        if i + 1 >= n:
            break
        count = struct.unpack('<H', data[i:i+2])[0]
        i += 2
        
        if count & 0x8000:
            # Literal sequence
            literal_count = count & 0x7FFF
            for _ in range(literal_count):
                if i + 1 >= n:
                    break
                low = data[i]
                high = data[i + 1]
                i += 2
                result.append((low, high))
                if len(result) >= output_words:
                    break
        else:
            # Run of same word
            if i + 1 >= n:
                break
            low = data[i]
            high = data[i + 1]
            i += 2
            for _ in range(count):
                result.append((low, high))
                if len(result) >= output_words:
                    break
    
    return result


def rle_decode_bytes(data, output_bytes):
    """Decode standard RLE data (byte-based) for regular text mode.
    
    Format: count byte followed by byte(s)
    - If bit 7 clear: repeat next byte 'count' times
    - If bit 7 set: output next (count & 0x7F) bytes literally
    """
    result = []
    i = 0
    n = len(data)
    
    while i < n and len(result) < output_bytes:
        count = data[i]
        i += 1
        
        if count & 0x80:
            # Literal sequence
            literal_count = count & 0x7F
            for _ in range(literal_count):
                if i >= n:
                    break
                result.append(data[i])
                i += 1
                if len(result) >= output_bytes:
                    break
        else:
            # Run of same byte
            if i >= n:
                break
            byte_val = data[i]
            i += 1
            for _ in range(count):
                result.append(byte_val)
                if len(result) >= output_bytes:
                    break
    
    return bytes(result)


def render_memtext_frames(input_file, output_dir, progress_callback=None):
    """
    Parse and render frames from a MEMTEXT binary file.
    
    Args:
        input_file: Path to the .bin file
        output_dir: Directory to save output PNGs
        progress_callback: Optional callback(current, total, message)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'rb') as f:
        # Parse header
        header = parse_header(f)
        print(f"Header: {header}")
        
        columns = header['columns']
        rows = header['rows']
        mode = header['mode']
        
        if mode != 0x02:
            raise ValueError(f"Expected MEMTEXT mode (0x02), got {mode:02X}")
        
        palette_sets = [None, None]
        font_sets = [None, None, None, None]

        frame_idx = 0
        while True:
            chunk = parse_chunk_header(f)
            if chunk is None:
                break
            
            chunk_type, chunk_id, chunk_length = chunk

            if chunk_type == 0x01:
                if chunk_length == 2048:
                    lut_data = f.read(chunk_length)
                    palette = []
                    for i in range(256):
                        offset = i * 4
                        b, g, r, a = lut_data[offset:offset+4]
                        palette.append((r, g, b, a))
                    palette_sets[chunk_id & 0x01] = palette
                    print(f"Loaded MEMTEXT palette (LUT ID {chunk_id})")
                elif chunk_length == 128:
                    lut_data = f.read(chunk_length)
                    palette = []
                    for i in range(32):
                        offset = i * 4
                        b, g, r, a = lut_data[offset:offset+4]
                        palette.append((r, g, b, a))
                    while len(palette) < 256:
                        palette.append((0, 0, 0, 255))
                    palette_sets[0] = palette
                    palette_sets[1] = palette
                    print(f"Loaded standard palette (LUT ID {chunk_id})")
                else:
                    print(f"Warning: Unexpected LUT length {chunk_length}, skipping")
                    f.read(chunk_length)
                continue

            if chunk_type == 0x02:
                font_data = f.read(chunk_length)
                if chunk_length == 2048 and chunk_id < 4:
                    patterns = np.frombuffer(font_data, dtype=np.uint8).reshape(256, 8)
                    font_sets[chunk_id] = patterns
                    print(f"Loaded font set {chunk_id} (256 patterns)")
                else:
                    print(f"Warning: Unexpected font length {chunk_length} for set {chunk_id}")
                continue

            if chunk_type == 0x07:
                rle_data = f.read(chunk_length)
                if chunk_id < 4:
                    font_data = rle_decode_bytes(rle_data, 2048)
                    if len(font_data) == 2048:
                        patterns = np.frombuffer(font_data, dtype=np.uint8).reshape(256, 8)
                        font_sets[chunk_id] = patterns
                        print(f"Loaded RLE font set {chunk_id} (256 patterns)")
                continue
            
            if chunk_type != 0x00:
                print(f"Skipping unknown chunk type {chunk_type:02X}")
                f.read(chunk_length)
                continue

            is_memtext = (mode == 0x02)
            
            # Frame Start
            print(f"Processing frame {frame_idx}...")
            
            # Collect frame data
            char_data = []
            color_data = []
            while True:
                chunk = parse_chunk_header(f)
                if chunk is None:
                    break
                
                chunk_type, chunk_id, chunk_length = chunk
                
                if chunk_type == 0xFF:
                    # Frame End
                    break

                elif chunk_type == 0x01:
                    if chunk_length == 2048:
                        lut_data = f.read(chunk_length)
                        palette = []
                        for i in range(256):
                            offset = i * 4
                            b, g, r, a = lut_data[offset:offset+4]
                            palette.append((r, g, b, a))
                        palette_sets[chunk_id & 0x01] = palette
                    elif chunk_length == 128:
                        lut_data = f.read(chunk_length)
                        palette = []
                        for i in range(32):
                            offset = i * 4
                            b, g, r, a = lut_data[offset:offset+4]
                            palette.append((r, g, b, a))
                        while len(palette) < 256:
                            palette.append((0, 0, 0, 255))
                        palette_sets[0] = palette
                        palette_sets[1] = palette
                    else:
                        f.read(chunk_length)
                
                elif chunk_type == 0x02:
                    font_data = f.read(chunk_length)
                    if chunk_length == 2048 and chunk_id < 4:
                        patterns = np.frombuffer(font_data, dtype=np.uint8).reshape(256, 8)
                        font_sets[chunk_id] = patterns
                
                elif chunk_type == 0x07:
                    rle_data = f.read(chunk_length)
                    if chunk_id < 4:
                        font_data = rle_decode_bytes(rle_data, 2048)
                        if len(font_data) == 2048:
                            patterns = np.frombuffer(font_data, dtype=np.uint8).reshape(256, 8)
                            font_sets[chunk_id] = patterns
                
                elif chunk_type == 0x03:
                    # Fixed Frame Character
                    data = f.read(chunk_length)
                    char_data.append(data)
                
                elif chunk_type == 0x04:
                    # Fixed Frame Color
                    data = f.read(chunk_length)
                    color_data.append(data)
                
                elif chunk_type == 0x05:
                    # RLE Frame Character
                    rle_data = f.read(chunk_length)
                    if is_memtext:
                        chunk_words = 704 if chunk_id >= 4 else 1024
                        decoded = rle_decode_memtext(rle_data, chunk_words)
                        # Convert back to bytes
                        data = bytes([b for w in decoded for b in w])
                    else:
                        # Byte-based RLE
                        data = rle_decode_bytes(rle_data, 4800)
                    char_data.append(data)
                
                elif chunk_type == 0x06:
                    # RLE Frame Color
                    rle_data = f.read(chunk_length)
                    if is_memtext:
                        chunk_words = 704 if chunk_id >= 4 else 1024
                        decoded = rle_decode_memtext(rle_data, chunk_words)
                        data = bytes([b for w in decoded for b in w])
                    else:
                        data = rle_decode_bytes(rle_data, 4800)
                    color_data.append(data)
                
                else:
                    # Skip unknown
                    f.read(chunk_length)
            
            active_fonts = font_sets
            
            # Combine chunks
            all_char = b''.join(char_data)
            all_color = b''.join(color_data)

            if palette_sets[0] is None and palette_sets[1] is None:
                raise ValueError(f"No palette available while rendering frame {frame_idx}")
            
            # Render frame
            output_path = os.path.join(output_dir, f'reconstructed_{frame_idx:04d}.png')
            
            if is_memtext:
                render_memtext_frame(all_char, all_color, active_fonts, palette_sets,
                                    columns, rows, output_path)
            else:
                palette = palette_sets[0] if palette_sets[0] is not None else palette_sets[1]
                render_standard_text_frame(all_char, all_color, active_fonts[0], palette,
                                          columns, rows, output_path)
            
            print(f"Saved {output_path}")
            frame_idx += 1
            
            if progress_callback:
                progress_callback(frame_idx, -1, f"Rendered frame {frame_idx}")
        
        print(f"Done. {frame_idx} frames rendered.")
        return frame_idx


def render_memtext_frame(char_data, color_data, font_sets, palette_sets, columns, rows, output_path):
    """Render a single MEMTEXT frame.
    
    char_data: bytes with 2 bytes per tile
               Per spec: low byte [7:0] = CHARACTER, high byte bits:
               [8:9]=FONT BANK, [10:11]=combined LUT select, [12]=INVERT
    color_data: bytes with 2 bytes per tile: [bg_idx, fg_idx]
               Per spec: low byte [7:0] = BACKGROUND, high byte [15:8] = FOREGROUND
    font_sets: list of 4 numpy arrays, each (256, 8)
    palette_sets: list with LUT set 0 and 1 (each list of 256 (r, g, b, a) tuples)
    """
    cell_size = 8
    img_width = columns * cell_size
    img_height = rows * cell_size
    
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    n_tiles = columns * rows
    
    for tile_idx in range(n_tiles):
        # Parse character data
        if tile_idx * 2 + 1 >= len(char_data):
            break
        char_idx = char_data[tile_idx * 2]
        info_byte = char_data[tile_idx * 2 + 1]
        font_set_id = info_byte & 0x03          # bits [0:1] of high byte (bits 8-9)
        lut_sel = (info_byte >> 2) & 0x03       # bits [2:3] of high byte (bits 10-11)
        invert = (info_byte >> 4) & 0x01        # bit 4 of high byte (bit 12)
        
        # Parse color data: bg in low byte, fg in high byte
        if tile_idx * 2 + 1 >= len(color_data):
            break
        bg_idx = color_data[tile_idx * 2]       # Low byte = background
        fg_idx = color_data[tile_idx * 2 + 1]   # High byte = foreground
        
        # Get font pattern
        if font_sets[font_set_id] is None:
            # Fall back to font set 0
            font_set_id = 0
        if font_sets[font_set_id] is None:
            # No fonts available, skip
            continue
        
        pattern = font_sets[font_set_id][char_idx]
        
        # Use combined LUT selector for both FG and BG.
        # Valid selectors are 0 (00) and 1 (01). Any other value falls back to LUT0.
        if lut_sel > 1:
            lut_sel = 0
        fg_palette = palette_sets[lut_sel]
        if fg_palette is None:
            fg_palette = palette_sets[1 - lut_sel]
        bg_palette = palette_sets[lut_sel]
        if bg_palette is None:
            bg_palette = palette_sets[1 - lut_sel]
        if fg_palette is None or bg_palette is None:
            continue

        fg_color = fg_palette[fg_idx][:3]
        bg_color = bg_palette[bg_idx][:3]
        
        # Calculate tile position
        tile_row = tile_idx // columns
        tile_col = tile_idx % columns
        y0 = tile_row * cell_size
        x0 = tile_col * cell_size
        
        # Render tile
        for row in range(8):
            byte = pattern[row]
            for col in range(8):
                bit = (byte >> (7 - col)) & 1
                if invert:
                    bit = 1 - bit
                if bit:
                    img[y0 + row, x0 + col] = fg_color
                else:
                    img[y0 + row, x0 + col] = bg_color
    
    # Save image
    pil_img = Image.fromarray(img, 'RGB')
    pil_img.save(output_path)


def render_standard_text_frame(char_data, color_data, font, palette, columns, rows, output_path):
    """Render a standard text mode frame (16-color fg/bg)."""
    cell_size = 8
    img_width = columns * cell_size
    img_height = rows * cell_size
    
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    n_tiles = columns * rows
    
    for tile_idx in range(n_tiles):
        if tile_idx >= len(char_data) or tile_idx >= len(color_data):
            break
        
        char_idx = char_data[tile_idx]
        color_byte = color_data[tile_idx]
        fg_idx = (color_byte >> 4) & 0x0F
        bg_idx = color_byte & 0x0F
        
        # Get font pattern
        if font is None:
            continue
        pattern = font[char_idx]
        
        # Get colors (first 16 are FG, next 16 are BG in standard mode)
        fg_color = palette[fg_idx][:3]
        bg_color = palette[16 + bg_idx][:3]
        
        # Calculate tile position
        tile_row = tile_idx // columns
        tile_col = tile_idx % columns
        y0 = tile_row * cell_size
        x0 = tile_col * cell_size
        
        # Render tile
        for row in range(8):
            byte = pattern[row]
            for col in range(8):
                bit = (byte >> (7 - col)) & 1
                if bit:
                    img[y0 + row, x0 + col] = fg_color
                else:
                    img[y0 + row, x0 + col] = bg_color
    
    pil_img = Image.fromarray(img, 'RGB')
    pil_img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render MEMTEXT binary animation files to PNG frames")
    parser.add_argument('input', help='Input binary file (.bin)')
    parser.add_argument('output_dir', help='Output directory for PNG frames')
    args = parser.parse_args()
    
    render_memtext_frames(args.input, args.output_dir)


if __name__ == "__main__":
    main()
