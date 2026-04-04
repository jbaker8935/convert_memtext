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

# Global configuration flags set by command-line arguments
CROP_TO_FILL = False  # when True, images are scaled and then center-cropped to exactly 640×480
# Medoid derivation method (can be overridden from CLI)
MEDOID_METHOD = 'legacy'  # choices: 'legacy', 'oklab_kmeans'
MEDOID_CANDIDATES = 1
# (CLARA implementation removed; medoid method choices limited)

# Optional: prefer cuML KMeans if available (GPU acceleration). Fallback to sklearn KMeans.
try:
    from cuml.cluster import KMeans as cuKMeans
    _USE_CUML = True
except Exception:
    cuKMeans = None
    _USE_CUML = False

# Optional: use Numba for JIT-compiled high-quality tile matching.
try:
    import numba
    _USE_NUMBA = True
except Exception:
    numba = None
    _USE_NUMBA = False


def palette_list_index_to_chunk_id(idx, fg_count):
    """Map an index within a combined FG+BG palette list to a chunk ID.

    The combined list is expected to be [fg0, fg1, ..., bg0, bg1, ...].
    FG entries are assigned chunk IDs starting at 0; BG entries start at 2.
    """
    if idx < fg_count:
        return int(idx)
    return 0x02 + int(idx - fg_count)


def lut_id_to_fg_bg_chunk_ids(lut_id):
    """Return (fg_chunk_id, bg_chunk_id) for a given lut_id (0/1).

    This implements the double-buffering convention: fg = (lut_id & 1),
    bg = 2 + (lut_id & 1).
    """
    sel = int(lut_id & 0x01)
    return sel, 0x02 + sel

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

# Embedded default medoids (from medoids.npy).  This is the canonical 1024×8 medoid set
# used when the --default-medoids flag is supplied.
#
# NOTE: This is generated from the current medoids.npy in the repo and is intentionally
# self-contained so convert_memtext.py can run without any external files.
EMBEDDED_DEFAULT_MEDOIDS_B64 = ("""
AAAAAAAAAAA8fHh4eHh4eAMHBw9/fw8PAICAgIf///8fDwAAgP///wCAh4cPHz//AA///38HAwADQ8PD
x8fHhz9/fvjw4IAAfwcHDx8fHz8/f/8PBwMDAAAAYPP/Dx9/A8PDx8eHBwd8eHj4+PDw4ABwcPj4+Pz+
f38/HwSAwOAHDz8/Dw8HB/j4+IDg8Pz8Hh48PHh48PAPDx8/f/DAAD8/P3744AAAHx8PBwcHX/9/f3/+
eAAAADwAAODw/P7/////H3zAAAD////+8AAAAADA8Hh+fz8PAAAA/39/Pw98PDw4eHh4eADg4Pj4kICA
D/8PDwcHBwc8PDh4+PDw8PDwAADw////APD+/38AAH4AAP7/AID//38/AACA8Pz/AICEiQcX//8AxNg4
fPz+/ADgAgH/fz//ODj48PDw+Ph/PweB8fjwAAAAfwcDD///Pz8/f35wAAAAAAF/////AD8/eMAADPz8
8eHPjhwYOHB/f39+AAAA8AAHH////wAAAAAAB/////8AAf8PBwcf//78+Pjw0ICAAwODg8PH5/cAAAAA
Bv///wADf////wAAAAABZ08ff/8/f///gwAAAB8fPz9/f39//vCAAAEf//8BB39/fx8HA////z8HAAAA
fnw8vNAYMR4AAOD4AP///////8AAAAD/fz8fDwfDwQAfD4+Hx+Ph8X8/Px8PBwEAAIDwPj8/fzgBB/9/
HwcHAwDAgzc7v5+fAADA+P78//8AAID8/wP//w888MMffOCAAP8HAP//HwAf/x8f/wAAAD8fHw8PBwcD
P3//+ACAwPB/Hx9/fwEA/ADwAP//APj/AAD//48AB/8BAwcPDx//Hz8/f/8PAQAAAAAABx////8A///8
APjwgD8fHw8Hg4PBAPj8/Pz48AAAAPj//39wAP//AwAABx//AAA+///+4IAAAAD//////wCf////BwAA
Pz/fHHj4wAAAAAf//x8PDwAAAP8PH3//H///DwcDAQEAAODw+Pz+/wD+Af///wAAAPD8/PjgAP8AAADw
/P7//wCA4PB4Pn//AID4AP///wAAwIDg+Pz+/zDAAD////iAAAAAB///P38AgODw+Bz//38fz/74YAAA
f38HAP//AAA/AAAAAP///wAHPz8/Pz8/AAD7B/8HB/8AwP//Hw8HAwB/f38PBwMHAAB4/z9/fz8BAwcP
Dw8f/wD/Af//AAB/fz8BAADD//8/f//8+PAAAAMHBw8f/w8HAAD//wP//wAAAAEDDz///xzswADDgz9/
fzvwAA/+8IAA//8DAAAP////+AAAAP//Hhw8/Pz48AAA4ODAAP////j8+PjwMHDgAP8HAAD//w94OBiY
mMz8/gMPf///BwAAANifBwEHP/8Bf38PDxD/AAH/PwMH/w8ABwcHD/8PDwcAAP//Bwf/AwAf/w9//wAA
///8wAAAwP8AAP+PAP/z7/j4+nd2YWEBAAADBx9///9/AQAA//8D/wAHf3//fwAAAPj///wAAPh/fz8f
BwHgAAAH//8f/wAAPx8Pj4/DgAAA4P8A//8A/z8PBw///wAAfwOHwwAA+/8AgODw/Px+Pz+/n4mAwMDg
B4eHg8PD4+M/Hw8PDw8Pf/j48PDw8PDwfwAA////AAB/PwcBAQcB/wAAAP8HH///AAf//wAAB/8AAABg
/////z8///8A8AAA+Pj4uHBw8PD//39/PwEAAH8D//8AAAD/AAAA//+/P/8AAAMHDx///z8/Pz9/fwAA
AAAf////AwEAAAAP////AAB////zAOAAAMF/BwcPf38A//9/HwcAABg4ePj48PDwAwePj4+Pj48PAH8B
/wD/D////gAE/PAAAODwAID///8AAPgAAP///z8/Pz8fDwMADz////8AAADw+PDw8PD4+AD4+AD8//8A
AADvDw8fPz8AAAAAfn///3h4ePjw8PDwHhw8eHj48PAAAwcff//jgz5+/PjwgIDAPx8//wAAA/////8f
/gAAAAABBwcf//8PAAAA/x8f//8A+P9/HwMA4Dw4AAAA////AAAAAP////8fDwAA/w8D/398fHx4eDAA
Dw8f/x8HAwEA/AADDx9//////x8A+AAAfHx4ePjg4IAAAPz///+gAP//Hxz8AP8AAAAA//8HD/9/Hw8H
AwOD8////vzwwAAAAPj8/Pjw4MAAAPgA//8H/wAA+P///vAAAIDggA9///8/Px8fDwcBADicnM7nczEY
AAAAAf8f//8AAAD+gH///wAAAP//H/8AAICAwOf/fz///PjAAAD4/z8/Pz8AAAD/AIGBh4+Pn/8/BwEB
Awf//z9//wAAAAf/Awf//w8HAwM/P378+OAAAH8fD///AAAA/vz8+PDwwAB8eAAAAPr//wD///8AAH8A
4ePDw8eHh4c/P38PDwcDAD+f34MDAILnf39//wEAAAB/fwAAAAD//wAAAw8ff3//AAAA/AP///8+AAAA
8P///wAAfgcH//8PAP////8AAAA/Pz788MDAgD44ePjw8PDwAP8f/wcAAP8PHz9/fvCAAD8AAODwPH//
AODg4ICA//8AAAAA+P///x8fH39/DwcDAAD///8AAP94AADg+Pz8/wAA////PwcBAP//H/8DAAD/////
8AAAAAD///8G+AAAAP4AAAN///8DAwcPDw9/f3AA8Pj4+Pj4fwAA7gAA//8A+P////gAAH8/Dw//BwAA
Px8fDw8HAwP//vz48ODAgH9/Pw8HDx9/PHj4gAD8/Pw+fvgAAAD//39/AwAA//8AAPAA/v//+AAPf39/
fwcHAAAMf3///oAAAIDg+Px+Px9/B/8HAfgfAHh4eHh4+PjwD///AAAH/wABAf8HBw//DwcH/x8PBwcH
fx8DAID//wd+AAAA8P///wAA8wMDf///AID4+Pj48IADAwcPH38fDz8/fHj44MCAAAAHDw8Pf/////zg
gAAA/wAAb28fP38POHz8wAAA//8/Hx8PD/8AAD8fDw+Hg8PhAADw/P7+gID4+PAAAPD//wABAwcPH3//
AP///wAA/wD///8PAAAceDwA//8A8P8A//8fDwcHAwH//384+NAAAD8HAAD///8AAP//AAD//wB/Hw8H
BwMH/wAA8P74AP//AQMDBwcP//8/P///DwAAAAMDB4+Pj4+PfwPg8Pj4+AA/AAD4//8B8P/4wADgAP//
APD4+PgA+P94eHh4eHh4+AABBx///w8DAABr/39/PwAAgP7/AAD//wABh4ePD5//AAFjx48fP38BAfPf
BwE/fwDA/////wAA//CDH/CDPOAAB////w8AADw8eHh4eHjwOADgAPL8//8A////Dw8AAD9//vjAAAD4
AIDg/38PB+cfHw8PDw8HBwCAwODw/P//AYOHh4+Pj498fH7+8AAA+D8//wcE+AAAf39+/PDgAAA4eHh4
eHj4+AAAAf8PBx///////2AAAAA/Pz8/Pj4+PgD/Hz9/Hw4AAH9///8DAAB/H5//fwAAADg4+AAA8f//
AQMDBw8fP38AAID4/39/PwCAgID9////AAAA//8P//98fHx8ePDggD8+/Pjw4MCAB4eHh4ePjwcAf/8A
AAc//wCA//8A/v8AAP8AAP8P/x9/P/8/AQD4AP//Af8A/gMAB3//DwcHAwMAAQcff39/Hz8+fnx8fHx4
AP8f/wMA/Ad/Pw8PB+cDAH9/Pz8/Pzw8Dw+Hx8eHBwcA/////ICAAH8AAAD/A///AP//AP8A/wAI/wAA
B/8P/wD//wAAAX//Pn58+PDgwID//wMHD/8BAH9/fvz4gAAAAACAxD9/f/////8PBwMAAP///vBgYGAA
AQ9+4BzwB/9/fwAAgPD8/n8AAPj///AA/v78+PDAAOAAAABA7////x8fH//+YAAAAMDAAB////9/AAAA
AP///3h4eHh4eHh4AMDw/P7+/AD/+fDggODw8PzwAAAA////AAEDBwfP///////4AMDAAP////+AgAAA
AAD8AP8D//8AAID+///+AAAAAQ//Hz//////AAAAwP9/Hw+HweBwOAD//wAAAf//AADgAP////98AMDg
AP///wDAgID4/v//Pw8BAQ///wAAAP////wAAHh4ePj44GAAP3/++PDAgAB4eHh48PDw8Hh4eHw8OHh4
AP//Af//AAB/Hx8ffwMBAD8AgPz/AMD/AAABf38fH3////9/BwAAAAAA4OCA////AAABB/9/fz8AAP9/
AwMf//D/DwAAAz//fwD//wAHAPA/P//fjoAAAAMHf39/fwcHAPD/AACA//8A4ADw/P///38PAQABB///
AMDzBw8fPz8A/39/PwcBAQEHD3//DwcDAHj////wAAAAAADAx////wAAAACf////P3//nwBA4BAA////
AAAA/wDggACA////AADw8IA///////8fBwMAAADw//gAAP7/P38fDw8HAwMA/wD/AAD//wD8fwAAA///
/OAADPEHP/8A/wf/D/9UAAAAP3///8CAeHh4eHDw8PAAwcEBB/8fH3gAALj8A///fx8PDvz4gAAAAAAH
f3///38HAPgA//8Bfx8ff38BAAB/f3//AAAA4H8AAAD///8AAAAA/x///wB/f3/48ICAgD8P/wAA/wD/
AAAAAw9///8BB3//DwcHB38AAAN/f39/AQMHD3//HwN/PwCAwOD8/gACBw+fn5+fAJz//gAAH/8AAH//
//+AADw8PHh4ePDwAID4/39/DwAAAP///wD+AD8A8P4AAPz////4APj44AD///jgAAAHfwCA8Pj8nJz/
////j4TgAAAAMPj4+Pj4+AAPf38//wAA/////4AAAAAA8PB4fH5+fgAAAPz//wD/AwMHDx8fPz8/P358
+OCAAB8PDwcHh///AAEDP39//wH///34gAAA+Dh8/ICA8PD4fwAA////gAA/Hw8PDw9/fwDg4ICA////
f39+eAAA8PA4cOLGjQcfbwAAgPD4/v//AAAAB///H/8AAAAHf////z58/PDAwOD4AH9/f/8PAAB/fz8f
fwAAAAB//wAB//8AfwD/APD/APgAAA9//38PA38/Hw8Hf/4cAADwf39/P38AAwcPD3//Dz8/P39/DwcD
Pz///wMA4AA/f///AAAA//z48PDg8Pj4//8/Hx9/fw8A+Pz8/PiAAAD4//zggAL/fHx8+Hh4eGB4eHj4
+Pj4+DCw8PDw+Pj4APz//38HAAAAAP7/AP8B/z4+PDx8fHx4f39/f38AAAA/AID4//8A8D8//x8PAwEA
AAEPf///DwD8/PCAgOD4/wABAz////8BAAAA/////oAAAAf///+PAADw///wAID+AAB/Pw8HD/8A8PDH
h4+fnwAAgMeHH///AAAAD/////8/X28vP8DgAP///PjwwIAAP38DAQMDH/84ODj4+Pj4+H58fPjw4MCA
OHh4eHj4+Ph/f398+OAAAD8/f378fHx4/////PDAAAA/BwGAwD4//wAAB///Dw9/fwP4APcDAP8/BwAA
AP///wcHDw8PDw8PHx8fHwwAAP8/Pz58ePDAAH9/f38fAwAAPx8PBwcHf/8A////AAAA/wD4/wD4APz/
AAc/f3//AAAAAABjf3///z8/Hzn4+IAAPx8PD4fDYTB4cADg8Pj8/gAA//8BBH8///AH//7wAAAA/38H
f/8AAD8/f38ZAIDAAAADD///fwf44ODw+Pj4+ACAgIEHH///Hz8/Hw8HBwc+Pnz48PDwADw8OHh4+PDw
AAD//wMAf/8AfwEAHP//DwAD/x8f/w8A//8AAAP//wAAAH9/f39/Az8DAP///wAAAwcH/w8PDw8AAAcH
Dx9//z8/P3//AAAAAPj///9wAAD8/LCAAODw+ACAgIL/P39/AAAA/w//D/8AAAd/f38fB38PH/8AAAP/
APD4AAD+//8AgOADB3///z9/f//4AAAAAAAAh09/f/8/HwcHBwcf/wCA8PCQmv7/Hx8fHz9/fwB/HwAA
AP//DwEDB38fDw8PfgD4///4AAADA4+Pj4fHx38DAQAA/x//Px+AgMDg/P8AAAAA////////AAAAA///
AH///wAAA38/Pz9//IAAAP////DgwIAAAAD8///+AP8AAP8AB/8f/wCA4AB4///////+/PDAAAD/////
BwAAAP///PDggICAPz9///iAAAB/DwAA/wH/Hzh4eHDw8Pj4AQMHDw8PH38AgP8AAP///39/PweEgMDg
PH744MAA8P8Hb28PDwcHBwDA8PD4+Pz+AAEDD38fP38A+PjwAMD//wAAA////wcB/vz4+PDgwMAAgMAA
+P///xwcPDw8eHj8B/8f//8AAAAAAAB/Dx////j4+Pj4+BD8APz//wDw/AAAgICAj5///wAA/wADD///
P3//P3AwAAAAgOAAAP///zwDAAD///8AYHCwsPj4+Px/Pz//DwAAAD8/Hx//fwAAfwDw//8AAPg+AADw
/wD//wEHf/9/DwMAAODgAOD+//8Afx8D//8DAD8/fx8ff38PHx8f/x//AAA/P39/YAAA4AAAAAEP////
Px+f/38AAAAAwH5+fDw+/39/fz8H4AAAPx8fH38PAQADf///AQD4AD8/Pz8/P39/AID//38fBwD//38P
BwcBAADgAACA////AMDwAHB/f/9/f3//+AAAAH9/cAAAwPj/f38fBwEAA/8+PPz4+PDgAP////wAAPAA
OH//gACA8P8AAP//AwcffwAA+PjwgP//AADgcXl+f/98PHz8/PwA4AEDBw8PHz9/AACAgPz///9+fnx4
+PDgAP////AA4ADnPz8/P3zgAAB8ODg4eHh4/Hx8fDw8PDwcAIDg8AD///8AgICA+P///wABAw8PH3//
cHB4ePj4+PgAAwcPH39/fwf/fw8PBwMBAIDA8Pz//4MAAAD////+AH8AgIDA+P//AAAAf38fH/8AAAB/
/wc//wcHBwcHj4/PP39/+HAggMB/Pz5+fXx4IAAA/P//gYf/8PDwMDAw/v8+fPz48ODAgAAAAP//f38/
AAB///8P+AAAAPDw8AD//wD/AP//AAD///8AAAMP/x88fGAA8Pz8/AD///8HAAAA/Pz8YADw+P4AAP//
/4cA/3h4eHj4eHh4AQMHDw8PH38A//8AAP8A/z8/f38PAwAAAABgf39/f/8AAAN/f///AAAAAP///4D/
fz8fD4fDgQAAAAD/f39//38fHw8PBwcDAwMHDw8ffx9/fzwAMHh4eB8+fPjw4MCA/vz8+PDw4AB/D//8
wADwAP/8+IAAAPP/AAAA8H8///8A//8PBwcHBz9+fAAAgP7+AP8/f/8HAAA/Hw8HfPjwAAAAgICc////
eAABBw8/f/8fHx8/Pz8/PwB/f/8/BwMBP39/Hx8/P38AAOD4PF5//38fDw8HB/MAPj58/PDgAOAAAPCA
AP///wDgAPgA////fwAA//8DAP8/f3//fwAAAAD+//8AAID/f38/PwcBAAAAAP7/AQD//wAAgOD/////
AP//fgAAA/8AAwcf/38PDwAAAH//////AAD/A///fwc+Pjw8PHz8/ACBg4OHD///+Pj4+PBwcHAABx9/
//+AAAAA/4cDB3//AIDg8Pj+//8A8AAAB////wAAA/8fHz8/Px8PB4PB4fAAAHj/AA///z8/P31wYOAA
fw8H/wEA/wM/Pz//HwMAAD8DAP//AAD/AAAAPP7////4+Ph4eHh4cACAwP///z8AAAN7Aw8f//8/B/8A
AAD//wDDxwcHBz//AHD/H///AAAA4AD/A/8H/wAB/wAf/38PAP4AAP//Af8A4eH////AAH8/D4OAgMD/
AAD+//8AgP8AZ0cHBw8f//jwwAD4/Pz4//8AAAAA//8AAACDn////39/Px8PBwEAfw8PDw8PZ28AAH//
/wAD/z8/f/7YQIAAfwAAgPD4/v8/AAAA/wP//wEPz8/PBwcHAQcPf38PBwd/D///AADwAADA/39/DwcB
Pz8PAAAA//8AcHA9H///AAAAAP+fHz///Pg4PHjw/IgAAQMPP/9/Hw8/f///wAAAAICAgAf///8AAP//
///AAADwgAAA////APj4+Pj4gIAAQP/8/OAA/wDAIBh//z9/Hx8/fx8HAQAAAOGBBw////j48PDg4Pj+
AAAHH/8PH38ABx8PB3//AAAA8Px+f39/Pw8AgOD4/P5/fHh4ePDwAAAAAw8/P///AAB4f/9/f39+Pj4+
Pj4+Pgd/Px8f/x8/fwcHD///AAAfHw8PBwf/Dz8PAQABh///AMD8AT///4AAAP8A/wD/AD8////8AAAA
Pz8PD39/fwABBx8/Pz9/fwB//38PBwMBBw8fHz/nwID8/Pj48IDg8AD4fw///wAAP3//fwAAAPgAAAD/
//8D////jwcDAwd/fw///wD4AAAAAAf//38PAwABAwMPH///AHh/Pz8/Px8AAP8PD///AH8H/gD//wAA
AABgf3///wB/BwAAAP///z8fDwMD/38P////AAAAAP8A/P4A/H8A/AAAAH///x//4OBgeBkd//8AAPz/
f38PAD8fj8djEZjEAADgeH8/Pz8DP/8AAAF//wAABwcPf38/PDw8fHx4eHw+fgAAwPD//zh4+PDw8PD4
OHh4+PDw8PB/Px748ODAgH58fHx4eHAAPwAA+P//+AAAAID4/z9//wCAgPz///8AAP7//4AAAP9/HweH
9/MAAP///wAAAA///vz4+PDgwMB/BwDAOM89839/f3xgAADwPwAA/wAA//8/D4fDw2A4HAAAcH////kA
AAD8/v744MB/A/8AAAD//z9+fHx4cHAA////AwH+AAD///8AAPzwAACA////AAD/AX8fBw8//w8/H/8H
AQAE/3x4AADw+Pz/AICAx/f3x8cAv/8D/wAA/gAAAID4////AANf/x8f/AAAg4eHDx8/PwAAgICfn///
APwAAP///wAAgIDg/59//////QAAcHD4AP//AAD/fwAHf///DwMAAAA/////gAAAA4ePj4+PBwcBAcOH
Dx8/fx4IAPD4+Pz+/////w8AAAAAAODcfj8//wDA////PwAAZjEchuPw+P4cPDx4+PDw8AAA//8fPz8/
fxMAAAN/f38AAID///8A/wADBw8ff38fAP//Bwf/AAAAMDHPjw8fHwDw+Pj4gMD8fwAA//8B/wMAAAAP
f////z4+fnx48MCA/Pj48PDw8PAA//9/BwH4AD8fH///AAAAAAADD3///wcA/v+AAAD//wAAAPj/f4D/
/////wAAAAD//+AA4AEf/wD/Dw//PwMAAD///vz4gAAAAP////8MAPz8+PhwcHDgPwAAAP7/B/9+AAAA
AP///////38HAQAAPwcPDwF/P39/f358fPAAAP////jAAAAA////8AAAAPz//wAP//MAADg4gODw+Pz+
////f34AAAB/Pz9/DwMAAB8fP39/fwAA///w4ACA8Px/fzsAAACP/wB//gAAA///AAF///9/AQB/fz8/
Hx9/f////w/8AAAAAAABAx9/f/9/HY2BgY+HhwCA4Ph8fH8/AP//AwEDPz8AAIOP/w8P/+HBw4eHDx8/
AwcHB8fHx8c/H4+Hg8PA4AAAwP//P39/AAP/D///AAA/AQAA//93Bz8fDw8PHx8fAIDw8Pj4/PwA4AAA
/P///wF7BwMDD///AwNvfw8PDwcAAQcHH3///38AAAAA////AICA4Pj8//8AgIDB/38/Px8/ffvgIGBg
APj4+IAA//8A+AAA/gP/////////////AAD8AP///wD///kAwIDA/j9+fHjw4MCAP39/Pw8DAAAAgMBg
eH///z8+/Pj4gADwHx8Pj4eHgwN8fAAAAH///wAAA39/f38Pfnx8+PDw4AAAAAB//39/fz8D/+AAwIb/
AP////gA4AAA4PBwcDj//38/Dw9/4QAAAAABD3////9/f3//BwAAAHgwkODw+Pz8f394eODgwMAAAB9/
Dx//Dz9/f3x8fAwA///gAAAA/////vz8wAAA8D9/fwMAAHj+fwD/AAD//wH///8A/PAAAAAB/39/HwcB
AwcPHz8fDwcAAw//fx8PBz9///7ggAAAPz9/f38AAAA/f3wAAAD7/wAA////fwAAAP//gP//AAD///z4
8ODgAD8fDwcHBw//AKDhQwcff/8AAAAAB////wD//wAAP/iPAP8AAAAP//8AAvzwgw9////4AAD///gA
////AAAAcH8AAAABfz8//zlhw4cHDx8/AAAH/z9//wB8fHzgADz8/ADA4MDA4P//AAAAf////////gAA
/ukD8wAAAwd/f3//fx////gAAAD8+Pj4eHBwcACAgP//Dz//AQcPDz9/bw8A/v8PH38BAP////6AgAAA
AP8A//8A+AADx38/PwQE/ADAeAD///8AAADg/P//gPw/Hx8f/wcAAAcPP///BwAAAPj/AID//wB/f39/
/AAAAAAHf//8wAx4AP//fw//AAB8PITc/Ph4MDwc/wAA//8AOJiYyOj4+Pw/f/9/DwAAAD9/////P8IA
fwAAIH9///8Df/8HD/8AAP///wMAAAH/P3//BwcHAAB/Hw8PB2NjAQAHf3///wAAAAADfx8fP/8/AAAA
Af///wD/DwMHBw///Pj4+PDw8OD//uEPfvCAAAAAAH9//38fAP8DAP8DD/8AgPD8/v8A/AM+AQf/DwD/
//sD//gAAAAAAP//fwP/AADBgwcHH3//f38PDwcHBwcAwOCA8Pz//wAAAw9/fx9/AAD+/////wAAAF//
Dx8ffz8///nBAQB//////wAAAAAfP3/88OCAAAAAf/8HBw//eHh4eHh4+PgA8Bj//wCg/wCAwAD/A///
AP0Hf39/f38/BwAABg///z9/HwcHPz8/AGD/fz788AA=
""" )


def load_default_medoids():
    """Return the embedded default medoid set (1024x8 uint8).

    Returns an array of shape (1024, 8) with dtype uint8.
    """
    import base64
    data = base64.b64decode(EMBEDDED_DEFAULT_MEDOIDS_B64)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 8)
    if arr.shape[0] != 1024:
        raise ValueError(f"Embedded default medoids have wrong shape {arr.shape}")

    # Ensure the zero-anchor invariant
    zero_pattern = np.zeros(8, dtype=np.uint8)
    if not np.array_equal(arr[0], zero_pattern):
        found = False
        for j in range(1, arr.shape[0]):
            if np.array_equal(arr[j], zero_pattern):
                arr[j] = arr[0].copy()
                arr[0] = zero_pattern
                found = True
                break
        if not found:
            arr[0] = zero_pattern

    return arr


def main():
    parser = argparse.ArgumentParser(description="K2 MEMTEXT animation converter")
    parser.add_argument('--animation', type=str, required=True, help='Input directory for animation frames')
    parser.add_argument('--output-bin', type=str, required=True, help='Output binary file (animation mode)')
    parser.add_argument('--frame-duration', type=int, default=6, help='Frame duration in 60Hz ticks')
    parser.add_argument('--rle', action='store_true', help='Enable RLE encoding for binary output (where supported)')
    parser.add_argument(
        '--encoding-mode',
        choices=['global', 'frame', 'hybrid'],
        default='global',
        help='Encoding mode: global uses one palette/font set for all frames, frame derives palette/fonts per frame, hybrid uses global palette with per-frame fonts. Modes frame/hybrid are experimental.'
    )
    parser.add_argument(
        '--coherence-medoids',
        type=int,
        default=0,
        help='Number of globally-stable coherence medoids to reserve for frame stability (0=disabled, e.g. 32, 64, 128). Only used with frame/hybrid modes.'
    )
    parser.add_argument(
        '--high-quality',
        action='store_true',
        help='Use reconstruction-error search for tile assignment. Evaluates multiple palette-color candidates and selects the combination that minimises pixel-level error. Slower but produces better visual quality.'
    )
    parser.add_argument(
        '--split-palette',
        action='store_true',
        help='Use separate 256-color palettes for foreground and background (512 total colors). '
             'Provides better color fidelity for most animations.'

    )
    parser.add_argument(
        '--denoise-medoids',
        type=int,
        default=0,
        metavar='SIZE',
        help='Remove pixel-island artifacts from medoid patterns. '
             'Connected components (8-connectivity) of set or clear bits with '
             'pixel count <= SIZE are flipped. 0=disabled (default), 1=single-pixel '
             'islands only, 2=1-and-2-pixel islands, etc. '
             'Typical values: 1 (conservative) or 2 (moderate).'
    )
    parser.add_argument(
        '--four-color-luts',
        action='store_true',
        help='Enable per-tile selection of 2 FG + 2 BG 256-color LUTs (uses bits 10/11 in char word)'
    )
    # Medoid tuning flags removed; medoid method is implied by --high-quality
    parser.add_argument(
        '--crop-to-fill',
        action='store_true',
        help='Scale and then center-crop input images so the 640×480 canvas is ' \
             'completely covered. Uses equal cropping on both sides of the ' \
             'overflowing dimension. By default images are padded with black.'
    )
    parser.add_argument(
        '--default-medoids',
        action='store_true',
        help='Use the embedded default medoid set (built into convert_memtext.py). '
             'When supplied, per-animation clustering is skipped and the embedded '
             '1024 medoids are used in global mode.'
    )
    args = parser.parse_args()

    # Frame encoding mode cannot be used with four-color LUTs because
    # palettes are double-buffered on the target; report an error to the user.
    if args.encoding_mode == 'frame' and args.four_color_luts:
        parser.error("--four-color-luts is not supported with --encoding-mode frame")

    # Set medoid method globally for helper functions
    # High-quality now uses the perceptual KMeans medoid derivation
    # (`derive_medoids_oklab_kmeans`) which is faster and perceptually similar.
    # Otherwise use legacy method.
    global MEDOID_METHOD
    MEDOID_METHOD = 'oklab_kmeans' if args.high_quality else 'legacy'

    # honour new cropping flag
    global CROP_TO_FILL
    if args.crop_to_fill:
        CROP_TO_FILL = True
        print("  Crop-to-fill: ON (canvas will be filled by cropping overflow)")

    print("K2 Memtext Image Converter")
    print("============================")
    print("Animation mode")
    print(f"Processing directory: {args.animation}")
    n_coherence = args.coherence_medoids
    high_quality = args.high_quality
    split_palette = args.split_palette
    denoise = args.denoise_medoids
    if high_quality:
        print("  High quality: ON (reconstruction-error tile matching)")
    if split_palette:
        print("  Split palette: ON (256 FG + 256 BG = 512 total colors)")
    if denoise > 0:
        print(f"  Denoise medoids: ON (removing islands <= {denoise} pixels, 8-connectivity)")
    if args.encoding_mode == 'global':
        if n_coherence > 0:
            print("Note: --coherence-medoids is ignored in global mode (all medoids are already global)")
        precomputed_medoids = None
        if args.default_medoids:
            print("  Using embedded default medoids (built into convert_memtext.py)")
            precomputed_medoids = load_default_medoids()
        print("MEMTEXT global mode: global palette, 1024 medoids in 4 font sets, shared across all frames")
        if precomputed_medoids is not None:
            print("  Medoid source: embedded default (skipping per-animation clustering)")
        frames_data = process_animation_frames_memtext_global(
            args.animation, high_quality=high_quality, split_palette=split_palette,
            denoise=denoise, medoids_arr=precomputed_medoids, four_color_luts=args.four_color_luts)
    elif args.encoding_mode == 'hybrid':
        if args.default_medoids:
            print("Warning: --default-medoids is only supported with --encoding-mode global; ignoring")
        print("MEMTEXT hybrid mode: global palette(s), per-frame 512 medoids in 2 font sets")
        if n_coherence > 0:
            print(f"  Coherence: reserving {n_coherence} of 512 medoid slots for cross-frame stability")
        frames_data = process_animation_frames_memtext_hybrid(
            args.animation, n_coherence=n_coherence, high_quality=high_quality, split_palette=split_palette,
            denoise=denoise, four_color_luts=args.four_color_luts)
    else:
        if args.default_medoids:
            print("Warning: --default-medoids is only supported with --encoding-mode global; ignoring")
        print("MEMTEXT frame mode: per-frame palette(s), 512 medoids in 2 font sets, double-buffered LUT/font IDs")
        if n_coherence > 0:
            print(f"  Coherence: reserving {n_coherence} of 512 medoid slots for cross-frame stability")
        frames_data = process_animation_frames_memtext_frame(
            args.animation, n_coherence=n_coherence, high_quality=high_quality, split_palette=split_palette,
            denoise=denoise)

    if not frames_data:
        print("Error: No frames processed successfully")
        sys.exit(1)

    print("Saving memtext animation output...")
    save_output_bin_memtext(
        frames_data,
        args.output_bin,
        frame_duration=args.frame_duration,
        use_rle=args.rle,
        encoding_mode=args.encoding_mode,
        four_color_luts=args.four_color_luts,
    )
    print("MEMTEXT animation conversion completed successfully!")


def load_and_scale_image(input_path):
    """Load and scale image to 640x480 with aspect ratio preservation.

    By default the scaled picture is padded with black to fit the 640×480
    canvas.  If :pydata:`CROP_TO_FILL` is True (set by ``--crop-to-fill``
    command‑line option) the image is instead scaled so the canvas is fully
    covered and the overflow dimension is cropped equally on both sides.
    """
    global CROP_TO_FILL
    img = Image.open(input_path).convert('RGB')
    target_size = (640, 480)

    # Calculate ratios for decision making
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if CROP_TO_FILL:
        # Scale so that the image completely fills the target area, then crop
        if img_ratio > target_ratio:
            # source is wider than target, scale height then crop width
            new_height = target_size[1]
            new_width = int(img_ratio * new_height)
        else:
            # source is taller (or equal), scale width then crop height
            new_width = target_size[0]
            new_height = int(new_width / img_ratio)

        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        # center-crop to target size
        left = (new_width - target_size[0]) // 2
        top = (new_height - target_size[1]) // 2
        right = left + target_size[0]
        bottom = top + target_size[1]
        return img_resized.crop((left, top, right, bottom))
    else:
        # original behaviour: scale to fit within target and pad with black
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


def rgb_to_oklab(rgb):
    """Convert RGB (uint8 0..255 or float 0..255) to Oklab (L,a,b).

    Vectorized NumPy implementation. Input shape (...,3). Returns same shape.
    """
    arr = rgb.astype(np.float32) / 255.0

    # linearize sRGB
    mask = arr <= 0.04045
    lin = np.where(mask, arr / 12.92, np.power((arr + 0.055) / 1.055, 2.4))

    # Linear transform to LMS
    l = 0.4122214708 * lin[..., 0] + 0.5363325363 * lin[..., 1] + 0.0514459929 * lin[..., 2]
    m = 0.2119034982 * lin[..., 0] + 0.6806995451 * lin[..., 1] + 0.1073969566 * lin[..., 2]
    s = 0.0883024619 * lin[..., 0] + 0.2817188376 * lin[..., 1] + 0.6299787005 * lin[..., 2]

    # cube root
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    lab = np.stack([L, a, b], axis=-1)
    return lab


def merge_palettes_for_medoids(fg_palettes, bg_palettes):
    """Merge multiple 256-entry palettes into a unique palette for medoid derivation.

    Skips palette index 0 (transparent) when present. Returns an (N,3) uint8 array
    of unique colors.
    """
    parts = []
    if fg_palettes is not None:
        for p in fg_palettes:
            if p is None:
                continue
            if p.shape[0] == 256:
                parts.append(p[1:])
            else:
                parts.append(p)
    if bg_palettes is not None:
        for p in bg_palettes:
            if p is None:
                continue
            if p.shape[0] == 256:
                parts.append(p[1:])
            else:
                parts.append(p)

    if not parts:
        return np.zeros((256, 3), dtype=np.uint8)

    merged = np.vstack(parts).astype(np.uint8)
    merged_unique = np.unique(merged, axis=0)
    return merged_unique


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
    print(f"Creating 256-color palette for MEMTEXT mode...")
    
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

    # Legacy behaviour: cluster on sampled pixels (preserve frequency weighting)
    # but guard against too few distinct colors.
    unique_pixels = np.unique(all_pixels, axis=0)
    print(f"  Found {len(unique_pixels)} unique colors")

    if len(all_pixels) < 255:
        # Not enough sampled pixels; fall back to unique colors
        n_clusters = min(255, len(unique_pixels))
        palette = np.zeros((256, 3), dtype=np.uint8)
        if n_clusters > 0:
            palette[1:n_clusters+1] = unique_pixels[:n_clusters].astype(np.uint8)
        for i in range(n_clusters+1, 256):
            palette[i] = [128, 128, 128]
    else:
        # Cluster on the sampled pixels (not the deduplicated set) to match
        # legacy behaviour where color frequency influences centroids.
        kmeans = make_kmeans(n_clusters=255, random_state=42, n_init=10)
        kmeans.fit(all_pixels)
        centers = kmeans.cluster_centers_.astype(np.uint8)

        palette = np.zeros((256, 3), dtype=np.uint8)
        # Index 0 reserved for transparency (black)
        palette[1:256] = centers
    
    print(f"  Created 256-color palette (index 0 = transparent/black)")
    return palette


def create_memtext_256_color_palette_from_image(image):
    """Create a 256-color palette from one scaled frame image."""
    img_np = np.array(image, dtype=np.uint8)
    pixels = img_np.reshape(-1, 3)
    # Cluster on per-pixel samples (legacy behaviour) but guard for too few
    # distinct colors in the single frame.
    pixels_f = pixels.astype(np.float32)
    unique_pixels = np.unique(pixels_f, axis=0)

    if len(pixels_f) < 255:
        n_clusters = min(255, len(unique_pixels))
        palette = np.zeros((256, 3), dtype=np.uint8)
        if n_clusters > 0:
            palette[1:n_clusters + 1] = unique_pixels[:n_clusters].astype(np.uint8)
        for i in range(n_clusters + 1, 256):
            palette[i] = [128, 128, 128]
    else:
        kmeans = make_kmeans(n_clusters=255, random_state=42, n_init=10)
        kmeans.fit(pixels_f)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[1:256] = centers

    return palette


def _split_tile_fg_bg_centers(img_np):
    """Compute per-tile FG/BG average colors using luminance median split.

    Args:
        img_np: (H, W, 3) float32 image array (must be 640×480).

    Returns:
        (fg_centers, bg_centers) – each a list of (3,) float32 arrays.
    """
    fg_centers = []
    bg_centers = []
    for by in range(0, 480, 8):
        for bx in range(0, 640, 8):
            tile = img_np[by:by + 8, bx:bx + 8].reshape(-1, 3)
            lum = 0.299 * tile[:, 0] + 0.587 * tile[:, 1] + 0.114 * tile[:, 2]
            threshold = np.median(lum)
            fg_mask = lum >= threshold
            bg_mask = ~fg_mask
            fg_centers.append(tile[fg_mask].mean(axis=0) if fg_mask.any() else tile.mean(axis=0))
            bg_centers.append(tile[bg_mask].mean(axis=0) if bg_mask.any() else tile.mean(axis=0))
    return fg_centers, bg_centers


def _cluster_centers_to_palette(centers_arr, label, seed=42):
    """Cluster a set of color centers to a 256-entry palette (index 0 reserved)."""
    unique = np.unique(centers_arr.astype(np.uint8), axis=0).astype(np.float32)
    n = min(255, len(unique))
    palette = np.zeros((256, 3), dtype=np.uint8)
    if n < 255:
        palette[1:n + 1] = unique[:n].astype(np.uint8)
        for i in range(n + 1, 256):
            palette[i] = [128, 128, 128]
    else:
        kmeans = make_kmeans(n_clusters=255, random_state=seed, n_init=10)
        kmeans.fit(unique)
        palette[1:256] = kmeans.cluster_centers_.astype(np.uint8)
    return palette


def _split_centers_two_palettes(centers_arr, label, seed=42):
    """Cluster into ~510 colors and split between two 256-entry LUT palettes.

    Runs KMeans with up to 510 clusters, then sorts the resulting centroids by
    luminance and splits them evenly: the darker half goes to LUT0 and the
    brighter half to LUT1.  This means each pair of LUTs together covers ~510
    distinct colors while remaining individually coherent, unlike two independent
    KMeans-255 runs which converge to nearly identical palettes.

    Returns (pal0, pal1) each (256, 3) uint8, index 0 reserved for transparency.
    """
    unique = np.unique(centers_arr.astype(np.uint8), axis=0).astype(np.float32)
    n_unique = len(unique)

    if n_unique <= 255:
        # Too few distinct colors — share a single palette across both LUTs
        single = _cluster_centers_to_palette(centers_arr, label, seed=seed)
        return single.copy(), single.copy()

    n_clusters = min(510, n_unique)
    print(f"  KMeans({n_clusters} colors) for {label} split across 2 LUTs...")
    kmeans = make_kmeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(unique)
    all_c = np.clip(np.round(kmeans.cluster_centers_), 0, 255).astype(np.uint8)

    # Sort by luminance so each LUT covers a contiguous brightness range.
    # Sort by luminance then interleave: even-indexed → LUT0, odd-indexed → LUT1.
    # Both LUTs cover the full luminance range with complementary colors, which is
    # better for gradients and avoids the "wrong-LUT" miss that happens when
    # the target color sits near the luminance boundary of a strict dark/bright split.
    # For n_clusters=510: each LUT gets exactly 255 entries.
    lum = (0.299 * all_c[:, 0].astype(np.float64)
           + 0.587 * all_c[:, 1].astype(np.float64)
           + 0.114 * all_c[:, 2].astype(np.float64))
    order = np.argsort(lum)
    all_c = all_c[order]  # sorted ascending by perceptual luminance

    # positions 0,2,4,... → LUT0;  positions 1,3,5,... → LUT1
    n0 = min(255, (n_clusters + 1) // 2)  # entries for LUT0
    n1 = min(255, n_clusters // 2)         # entries for LUT1

    pal0 = np.full((256, 3), 128, dtype=np.uint8)
    pal1 = np.full((256, 3), 128, dtype=np.uint8)
    pal0[0] = 0
    pal1[0] = 0
    pal0[1:n0 + 1] = all_c[0::2][:n0]
    pal1[1:n1 + 1] = all_c[1::2][:n1]

    print(f"    {label}: LUT0 {n0} + LUT1 {n1} interleaved colors (full luminance range in each LUT)")
    return pal0, pal1


def create_memtext_512_color_palette(image_files):
    """Create separate 256-color FG and BG palettes using role-aware clustering.

    For each 8×8 tile across all frames, pixels are split by luminance median
    into foreground (bright) and background (dark) groups.  The per-tile average
    of each group is collected, then clustered independently to produce
    role-specific palettes.  Index 0 is reserved for transparency in both.

    Returns: (fg_palette, bg_palette) each np.array of shape (256, 3), dtype=uint8
    """
    print("Creating 512-color split palette (256 FG + 256 BG)...")

    all_fg = []
    all_bg = []

    for idx, image_file in enumerate(image_files):
        img = load_and_scale_image(image_file)
        img_np = np.array(img, dtype=np.float32)
        fg_c, bg_c = _split_tile_fg_bg_centers(img_np)
        all_fg.extend(fg_c)
        all_bg.extend(bg_c)

    all_fg = np.array(all_fg, dtype=np.float32)
    all_bg = np.array(all_bg, dtype=np.float32)
    print(f"  Collected {len(all_fg)} FG centers and {len(all_bg)} BG centers from {len(image_files)} frames")

    fg_palette = _cluster_centers_to_palette(all_fg, "FG")
    bg_palette = _cluster_centers_to_palette(all_bg, "BG")

    print("  Created FG palette (255 colors + transparent) and BG palette (255 colors + transparent)")
    return fg_palette, bg_palette


def create_memtext_512_color_palette_from_image(image):
    """Create separate 256-color FG and BG palettes from one scaled frame."""
    img_np = np.array(image, dtype=np.float32)
    fg_c, bg_c = _split_tile_fg_bg_centers(img_np)

    fg_palette = _cluster_centers_to_palette(np.array(fg_c, dtype=np.float32), "FG")
    bg_palette = _cluster_centers_to_palette(np.array(bg_c, dtype=np.float32), "BG")
    return fg_palette, bg_palette


def create_memtext_4lut_palettes(image_files, split_palette=False):
    """Create two FG palettes and two BG palettes from an animation set.

    Collects role-aware (luminance-split) per-tile FG/BG color centers from
    ALL frames, then runs a single KMeans clustering with 510 targets for each
    role.  The 510 resulting centroids are sorted by luminance and evenly split
    between LUT0 (darker half) and LUT1 (brighter half).  Together, each pair
    of LUTs provides ~510 distinct role-aware colors — approximately twice the
    coverage of the 2-LUT split-palette mode.

    The ``split_palette`` parameter is retained for API compatibility but is
    effectively always True: role-aware splitting is always applied.

    Returns (fg_palettes, bg_palettes) each a list [lut0, lut1] of (256,3) uint8.
    """
    print("Creating 4-LUT palettes (FG0/FG1, BG0/BG1) — role-aware, 510-color split...")
    if not image_files:
        raise ValueError("No image files provided for 4-LUT palette generation")

    all_fg: list = []
    all_bg: list = []
    for image_file in image_files:
        img = load_and_scale_image(image_file)
        img_np = np.array(img, dtype=np.float32)
        fg_c, bg_c = _split_tile_fg_bg_centers(img_np)
        all_fg.extend(fg_c)
        all_bg.extend(bg_c)

    all_fg_arr = np.array(all_fg, dtype=np.float32)
    all_bg_arr = np.array(all_bg, dtype=np.float32)
    print(f"  Collected {len(all_fg_arr)} FG centers and {len(all_bg_arr)} BG centers from {len(image_files)} frames")

    fg0, fg1 = _split_centers_two_palettes(all_fg_arr, "FG", seed=42)
    bg0, bg1 = _split_centers_two_palettes(all_bg_arr, "BG", seed=42)

    print("  Created FG0/FG1 (510-color FG split) and BG0/BG1 (510-color BG split)")
    return [fg0, fg1], [bg0, bg1]


def create_memtext_4lut_palettes_from_image(image, split_palette=False):
    """Create two FG and two BG palettes from a single scaled image.

    Collects per-tile luminance-split FG/BG centers, then uses a 510-center
    KMeans split (luminance-sorted) to produce two complementary palettes per
    role, consistent with the animation-sequence version.
    """
    img_np = np.array(image, dtype=np.float32)
    fg_c, bg_c = _split_tile_fg_bg_centers(img_np)

    all_fg_arr = np.array(fg_c, dtype=np.float32)
    all_bg_arr = np.array(bg_c, dtype=np.float32)

    fg0, fg1 = _split_centers_two_palettes(all_fg_arr, "FG", seed=42)
    bg0, bg1 = _split_centers_two_palettes(all_bg_arr, "BG", seed=42)

    return [fg0, fg1], [bg0, bg1]


def extract_frame_tiles(image):
    """Extract 8x8 MEMTEXT tiles and source pixels from a scaled frame image."""
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
        char_img = img_np[y0:y0 + 8, x0:x0 + 8, :]

        frame_tiles.append({
            'pixels': np.ascontiguousarray(char_img.astype(np.uint8))
        })

    return frame_tiles


def _medoid_feature_weights():
    """Return (edge_w, density_w, raw_w) feature-block scaling factors."""
    return 0.1, 0.1, 10.0


# ---------------------------------------------------------------------------
# Pixel-island denoising for medoid patterns
# ---------------------------------------------------------------------------

def _connected_components_8(mat):
    """Label connected components of True cells in an 8×8 boolean matrix (8-connectivity).

    Returns (labels, n_components) where labels is an 8×8 int array
    with values 0 (background / False) and 1..n_components for each component.
    """
    labels = np.zeros((8, 8), dtype=np.int32)
    current_label = 0

    for sr in range(8):
        for sc in range(8):
            if mat[sr, sc] and labels[sr, sc] == 0:
                # BFS flood-fill
                current_label += 1
                queue = [(sr, sc)]
                labels[sr, sc] = current_label
                while queue:
                    r, c = queue.pop()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                if mat[nr, nc] and labels[nr, nc] == 0:
                                    labels[nr, nc] = current_label
                                    queue.append((nr, nc))

    return labels, current_label


def denoise_medoid_pattern(pattern, max_island_size):
    """Remove pixel islands from an 8-byte medoid bit pattern.

    Connected components (8-connectivity) of set bits *or* clear bits that
    have total pixel count <= max_island_size are flipped.  This is
    equivalent to a binary area-opening followed by an area-closing.

    The all-zeros and all-ones patterns are never modified.

    Args:
        pattern: np.array of shape (8,), dtype=uint8 — 8 bytes encoding 8×8 bits.
        max_island_size: maximum component size to remove (e.g. 1 or 2).

    Returns:
        np.array of shape (8,), dtype=uint8 — cleaned pattern.
    """
    if max_island_size <= 0:
        return pattern

    # Quick check: skip all-0 and all-0xFF patterns
    total_bits = sum(bin(int(b)).count('1') for b in pattern)
    if total_bits == 0 or total_bits == 64:
        return pattern

    # Unpack to 8×8 boolean matrix
    mat = np.zeros((8, 8), dtype=bool)
    for row in range(8):
        byte = int(pattern[row])
        for col in range(8):
            mat[row, col] = bool((byte >> (7 - col)) & 1)

    changed = False

    # Pass 1: remove small islands of set bits (foreground islands)
    labels_fg, n_fg = _connected_components_8(mat)
    for lbl in range(1, n_fg + 1):
        size = int(np.sum(labels_fg == lbl))
        if size <= max_island_size:
            mat[labels_fg == lbl] = False
            changed = True

    # Pass 2: remove small islands of clear bits (background islands / holes)
    labels_bg, n_bg = _connected_components_8(~mat)
    for lbl in range(1, n_bg + 1):
        size = int(np.sum(labels_bg == lbl))
        if size <= max_island_size:
            mat[labels_bg == lbl] = True
            changed = True

    if not changed:
        return pattern

    # Repack to 8 bytes
    result = np.zeros(8, dtype=np.uint8)
    for row in range(8):
        byte = 0
        for col in range(8):
            if mat[row, col]:
                byte |= (1 << (7 - col))
        result[row] = byte

    return result


def denoise_medoid_set(medoids, max_island_size):
    """Apply pixel-island denoising to an entire medoid array.

    Args:
        medoids: np.array of shape (n, 8), dtype=uint8.
        max_island_size: maximum island size to remove (0 = no-op).

    Returns:
        np.array of shape (n, 8), dtype=uint8 — cleaned medoids.
    """
    if max_island_size <= 0:
        return medoids

    cleaned = medoids.copy()
    n_modified = 0
    for i in range(len(cleaned)):
        before = cleaned[i].copy()
        cleaned[i] = denoise_medoid_pattern(cleaned[i], max_island_size)
        if not np.array_equal(before, cleaned[i]):
            n_modified += 1

    if n_modified > 0:
        print(f"  Denoise: modified {n_modified}/{len(cleaned)} medoid patterns "
              f"(removed islands <= {max_island_size}px)")
    return cleaned


def derive_medoids_from_tiles(frame_tiles, n_medoids, palette, fg_palette=None, bg_palette=None):
    """Derive medoid patterns from frame tiles using canonicalized inversion-aware patterns.

    Args:
        frame_tiles: list of tile dicts with 'pixels' key.
        n_medoids: desired number of output medoid patterns.
        palette: (N, 3) uint8 reference palette (merged FG+BG for oklab methods).
        fg_palette: optional (N_fg, 3) uint8 foreground palette for split-palette scoring.
            May be a combined palette (e.g. both FG LUTs concatenated for 4LUT mode).
        bg_palette: optional (N_bg, 3) uint8 background palette for split-palette scoring.
            May be a combined palette (e.g. both BG LUTs concatenated for 4LUT mode).
    """
    unique_canonical = {}
    use_oklab = (MEDOID_METHOD != 'legacy')
    palette_oklab = None
    try:
        if use_oklab:
            palette_oklab = rgb_to_oklab(palette.astype(np.float32))
    except Exception:
        palette_oklab = None

    for tile in frame_tiles:
        font_def, _, _ = optimize_character_memtext(tile['pixels'], palette, palette_oklab=palette_oklab, use_oklab=use_oklab)
        canonical, _ = get_canonical_pattern(font_def)
        key = pattern_key(canonical)
        if key not in unique_canonical:
            unique_canonical[key] = canonical

    patterns_list = list(unique_canonical.values())
    zero_tile = np.zeros(8, dtype=np.uint8)
    zero_key = pattern_key(zero_tile)
    if zero_key not in unique_canonical:
        patterns_list.insert(0, zero_tile)

    patterns_arr = np.array(patterns_list, dtype=np.uint8)
    cluster_target = min(len(patterns_arr), n_medoids)

    if cluster_target < n_medoids:
        clustered_medoids = patterns_arr.copy()
        if len(clustered_medoids) < n_medoids:
            pad_count = n_medoids - len(clustered_medoids)
            pad = np.zeros((pad_count, 8), dtype=np.uint8)
            clustered_medoids = np.vstack([clustered_medoids, pad])
    else:
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

        edge_w, density_w, raw_w = _medoid_feature_weights()
        edge_feats_n = norm_feats(edge_feats) * edge_w
        density_feats_n = norm_feats(density_feats) * density_w
        raw_bytes_n = norm_feats(raw_bytes) * raw_w

        perceptual_descriptors = np.concatenate([edge_feats_n, density_feats_n, raw_bytes_n], axis=1)

        try:
            pca = PCA(n_components=min(perceptual_descriptors.shape))
            pca.fit(perceptual_descriptors)
            font_features = pca.transform(perceptual_descriptors)
        except Exception:
            font_features = perceptual_descriptors

        # Route to the requested perceptual medoid method
        if MEDOID_METHOD == 'oklab_kmeans':
            clustered_medoids = derive_medoids_oklab_kmeans(
                frame_tiles, cluster_target, palette,
                candidates=max(2, MEDOID_CANDIDATES),
                fg_palette=fg_palette, bg_palette=bg_palette)
        else:
            # Legacy medoid derivation: perceptual KMeans on structural descriptors
            kmeans = make_kmeans(n_clusters=cluster_target, random_state=42, n_init=10)
            try:
                kmeans.fit(font_features)
            except TypeError:
                kmeans.fit(font_features)
            cluster_labels = kmeans.labels_
            clustered_medoids = compute_medoids(patterns_arr, cluster_labels, cluster_target)

    if len(clustered_medoids) < n_medoids:
        pad_count = n_medoids - len(clustered_medoids)
        pad = np.zeros((pad_count, 8), dtype=np.uint8)
        clustered_medoids = np.vstack([clustered_medoids, pad])
    elif len(clustered_medoids) > n_medoids:
        clustered_medoids = clustered_medoids[:n_medoids]

    # Ensure medoid 0 is the all-zeros pattern (needed for flat tiles).
    # Flat tile assignments use font_set=0, char_idx=0 → medoid 0.
    zero_pattern = np.zeros(8, dtype=np.uint8)
    if not np.array_equal(clustered_medoids[0], zero_pattern):
        # Find existing zero medoid and swap it to position 0
        found = False
        for j in range(1, len(clustered_medoids)):
            if np.array_equal(clustered_medoids[j], zero_pattern):
                clustered_medoids[j] = clustered_medoids[0].copy()
                clustered_medoids[0] = zero_pattern
                found = True
                break
        if not found:
            # No zero medoid exists; replace the least-used one at index 0
            clustered_medoids[0] = zero_pattern

    return clustered_medoids


def extract_canonical_patterns(frame_tiles, palette):
    """Extract unique canonical patterns from frame tiles.

    Returns:
        dict mapping pattern_key -> canonical np.array pattern
    """
    unique_canonical = {}
    use_oklab = (MEDOID_METHOD != 'legacy')
    palette_oklab = None
    try:
        if use_oklab:
            palette_oklab = rgb_to_oklab(palette.astype(np.float32))
    except Exception:
        palette_oklab = None

    for tile in frame_tiles:
        font_def, _, _ = optimize_character_memtext(tile['pixels'], palette, palette_oklab=palette_oklab, use_oklab=use_oklab)
        canonical, _ = get_canonical_pattern(font_def)
        key = pattern_key(canonical)
        if key not in unique_canonical:
            unique_canonical[key] = canonical
    return unique_canonical


def cluster_patterns_to_medoids(patterns_arr, n_target):
    """Cluster an array of canonical patterns to n_target medoids.

    Uses the same PCA + KMeans + Hamming-distance medoid refinement pipeline
    as derive_medoids_from_tiles.

    Args:
        patterns_arr: np.array of shape (n_patterns, 8), dtype=uint8
        n_target: desired number of medoids

    Returns:
        np.array of shape (n_target, 8)
    """
    n_unique = patterns_arr.shape[0]

    if n_unique <= n_target:
        result = patterns_arr.copy()
        if len(result) < n_target:
            pad = np.zeros((n_target - len(result), 8), dtype=np.uint8)
            result = np.vstack([result, pad])
        return result

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

    edge_w, density_w, raw_w = _medoid_feature_weights()
    edge_feats_n = norm_feats(edge_feats) * edge_w
    density_feats_n = norm_feats(density_feats) * density_w
    raw_bytes_n = norm_feats(raw_bytes) * raw_w

    perceptual_descriptors = np.concatenate([edge_feats_n, density_feats_n, raw_bytes_n], axis=1)

    try:
        pca = PCA(n_components=min(perceptual_descriptors.shape))
        pca.fit(perceptual_descriptors)
        font_features = pca.transform(perceptual_descriptors)
    except Exception:
        font_features = perceptual_descriptors

    kmeans = make_kmeans(n_clusters=n_target, random_state=42, n_init=10)
    kmeans.fit(font_features)
    cluster_labels = kmeans.labels_
    return compute_medoids(patterns_arr, cluster_labels, n_target)


def _pattern_bytes_to_mask(pattern_bytes):
    """Convert 8-byte pattern to boolean mask of length 64 (row-major)."""
    mask = np.zeros(64, dtype=np.bool_)
    for r in range(8):
        byte = int(pattern_bytes[r])
        for c in range(8):
            mask[r * 8 + c] = ((byte >> (7 - c)) & 1) != 0
    return mask


def _build_perceptual_cluster_features(pixel_means_oklab, struct_feats=None, perceptual_weight=8.0):
    """Build per-pattern clustering feature matrix in perceptual (Oklab) space.

    Uses per-position Oklab color means as the primary feature so that KMeans
    groups patterns that *look* similar when rendered.  This aligns the
    clustering objective with the Oklab reconstruction-error scoring metric and
    consistently outperforms structural byte/edge features for medoid selection.

    Args:
        pixel_means_oklab: (n, 64, 3) float32 — per-position Oklab means.
        struct_feats: optional (n, m) float32 — pre-normalised structural features
            appended with weight 1.0 for tiebreaking.
        perceptual_weight: scaling applied to Oklab features (default 8.0).

    Returns:
        (n, ncols) float32 feature matrix ready for KMeans.
    """
    n = pixel_means_oklab.shape[0]
    perc = pixel_means_oklab.reshape(n, 192).astype(np.float32) * float(perceptual_weight)
    if struct_feats is not None:
        return np.concatenate([perc, struct_feats.astype(np.float32)], axis=1)
    return perc


def _build_nn_palette_kernel():
    """Build a Numba-JIT parallel nearest-palette lookup kernel (avoids big (n,p) allocation)."""
    @numba.njit(cache=True, parallel=True)
    def _nn_pal_idx(queries, pal, pal_norms):
        """Return nearest-palette index for each query using parallel Numba loops.

        Args:
            queries  : (n, 3) float32 — query colors.
            pal      : (p, 3) float32 — palette.
            pal_norms: (p,)   float32 — precomputed palette squared norms.

        Returns:
            (n,) int64 — index into pal for each query.
        """
        n = queries.shape[0]
        p = pal.shape[0]
        result = np.empty(n, dtype=np.int64)
        for i in numba.prange(n):
            q0 = queries[i, 0]
            q1 = queries[i, 1]
            q2 = queries[i, 2]
            qn = q0 * q0 + q1 * q1 + q2 * q2
            best_d = np.inf
            best_j = 0
            for j in range(p):
                d = qn + pal_norms[j] - 2.0 * (q0 * pal[j, 0] + q1 * pal[j, 1] + q2 * pal[j, 2])
                if d < best_d:
                    best_d = d
                    best_j = j
            result[i] = best_j
        return result
    return _nn_pal_idx


if _USE_NUMBA:
    _nn_pal_idx_jit = _build_nn_palette_kernel()
    # Warm-up: trigger JIT compilation now at module load time (avoids first-call latency)
    _dummy_q = np.zeros((4, 3), dtype=np.float32)
    _dummy_p = np.zeros((4, 3), dtype=np.float32)
    _dummy_n = np.zeros(4, dtype=np.float32)
    _nn_pal_idx_jit(_dummy_q, _dummy_p, _dummy_n)
    del _dummy_q, _dummy_p, _dummy_n
else:
    _nn_pal_idx_jit = None


def _nn_palette_colors(means, palette_oklab, pal_norms=None):
    """Nearest-palette-color lookup via squared-distance BLAS matmul.

    Uses ||a - b||^2 = ||a||^2 - 2*(a·b) + ||b||^2 to avoid allocating a
    large (n, p, 3) broadcast temporary.  Much faster than np.linalg.norm
    for large n_eval × p comparisons.

    Args:
        means: (n, 3) float32 — query colors.
        palette_oklab: (p, 3) float32 — palette.
        pal_norms: (p,) float32 precomputed palette squared-norms, or None.

    Returns:
        (n, 3) float32 — nearest palette color for each query.
    """
    if pal_norms is None:
        pal_norms = (palette_oklab * palette_oklab).sum(axis=1)   # (p,)
    mean_norms = (means * means).sum(axis=1)                      # (n,)
    dots = means @ palette_oklab.T                                # (n, p) via BLAS
    dists_sq = mean_norms[:, None] + pal_norms[None, :] - 2.0 * dots
    return palette_oklab[np.argmin(dists_sq, axis=1)]             # (n, 3)


def _compute_cost_column_oklab(pattern_bytes, pixel_means_oklab_eval, palette_oklab,
                                fg_palette_oklab=None, bg_palette_oklab=None,
                                fg_pal_norms=None, bg_pal_norms=None):
    """Compute per-pattern Oklab reconstruction cost for one candidate medoid.

    For each pattern in the evaluation set, the candidate medoid's FG/BG mask
    is applied to the per-position Oklab means; the nearest palette color is
    chosen for each group and the total squared Oklab error is returned.

    Args:
        pattern_bytes: (8,) uint8 — candidate medoid.
        pixel_means_oklab_eval: (n_eval, 64, 3) float32.
        palette_oklab: (p, 3) float32 — fallback palette (both FG and BG).
        fg_palette_oklab: optional (p_fg, 3) float32 — FG palette.
        bg_palette_oklab: optional (p_bg, 3) float32 — BG palette.
        fg_pal_norms: optional (p_fg,) float32 precomputed FG palette squared-norms.
        bg_pal_norms: optional (p_bg,) float32 precomputed BG palette squared-norms.

    Returns:
        (n_eval,) float64 cost per pattern.
    """
    pal_fg = fg_palette_oklab if fg_palette_oklab is not None else palette_oklab
    pal_bg = bg_palette_oklab if bg_palette_oklab is not None else palette_oklab
    mask = _pattern_bytes_to_mask(pattern_bytes)
    fg_pos = mask
    bg_pos = ~mask
    n_eval = pixel_means_oklab_eval.shape[0]
    cost = np.zeros(n_eval, dtype=np.float64)

    if fg_pos.sum() > 0:
        fg_means = pixel_means_oklab_eval[:, fg_pos, :].mean(axis=1)   # (n_eval, 3)
        fg_colors = _nn_palette_colors(fg_means, pal_fg, fg_pal_norms)
        dif = pixel_means_oklab_eval[:, fg_pos, :] - fg_colors[:, None, :]
        cost += (dif * dif).sum(axis=(1, 2))

    if bg_pos.sum() > 0:
        bg_means = pixel_means_oklab_eval[:, bg_pos, :].mean(axis=1)
        bg_colors = _nn_palette_colors(bg_means, pal_bg, bg_pal_norms)
        dif = pixel_means_oklab_eval[:, bg_pos, :] - bg_colors[:, None, :]
        cost += (dif * dif).sum(axis=(1, 2))

    return cost


def _build_cost_matrix_batched(patterns_arr, eval_px, pal_fg, pal_bg,
                                fg_pal_norms=None, bg_pal_norms=None,
                                batch_size=32):
    """Build full (n_eval, k) cost matrix for all medoid patterns simultaneously.

    Vectorized replacement for looping _compute_cost_column_oklab k times.
    Uses BLAS matmul: (n_eval, 64) @ (64, k) to compute weighted position means
    for ALL k medoids at once, then computes nearest-palette in batches to
    keep peak memory bounded.

    The quadratic identity ||a - b||^2 = ||a||^2 - 2*(a·b) + ||b||^2 is used
    so the sum of squared deviations reduces to three BLAS/element-wise terms.

    Args:
        patterns_arr: (k, 8) uint8 — medoid patterns.
        eval_px: (n_eval, 64, 3) float32 — eval pattern pixel means in Oklab.
        pal_fg: (p_fg, 3) float32 — FG palette in Oklab.
        pal_bg: (p_bg, 3) float32 — BG palette in Oklab.
        fg_pal_norms: (p_fg,) float32 precomputed FG palette squared-norms, or None.
        bg_pal_norms: (p_bg,) float32 precomputed BG palette squared-norms, or None.
        batch_size: medoids per mini-batch for nearest-palette lookup.

    Returns:
        cost_matrix: (n_eval, k) float64.
    """
    k = len(patterns_arr)
    n_eval = eval_px.shape[0]

    # Precompute all FG/BG masks as float32 weight matrices (k, 64)
    all_masks_fg = np.array(
        [_pattern_bytes_to_mask(patterns_arr[j]) for j in range(k)],
        dtype=np.float32)
    all_masks_bg = 1.0 - all_masks_fg
    fg_counts = all_masks_fg.sum(axis=1)   # (k,) number of FG positions
    bg_counts = all_masks_bg.sum(axis=1)   # (k,)

    # Ensure float32 for BLAS; avoid copy if already float32
    eval_f32 = eval_px if eval_px.dtype == np.float32 else eval_px.astype(np.float32)

    # Sum-of-squared pixel values per position: (n_eval, 64) — computed once
    eval_sq = (eval_f32 * eval_f32).sum(axis=2)  # (n_eval, 64)

    # eval_sq_sum[i, j] = sum_p weight[j,p] * sum_c eval_px[i,p,c]^2
    #   = BLAS (n_eval, 64) @ (64, k)  →  (n_eval, k)
    eval_sq_sum_fg = (eval_sq @ all_masks_fg.T).astype(np.float64)   # (n_eval, k)
    eval_sq_sum_bg = (eval_sq @ all_masks_bg.T).astype(np.float64)   # (n_eval, k)

    # Per-channel summed pixel values over FG/BG positions:
    # fg_sums[i, j, c] = sum_p mask_fg[j,p] * eval_px[i, p, c]
    #   = BLAS (n_eval, 64) @ (64, k)  per channel  →  (n_eval, k)  ×3
    fg_safe = np.where(fg_counts > 0, fg_counts, 1.0).astype(np.float32)
    bg_safe = np.where(bg_counts > 0, bg_counts, 1.0).astype(np.float32)

    fg_sums = np.stack(
        [eval_f32[:, :, c] @ all_masks_fg.T for c in range(3)],
        axis=2)  # (n_eval, k, 3) — pixel sum over FG positions per medoid
    bg_sums = np.stack(
        [eval_f32[:, :, c] @ all_masks_bg.T for c in range(3)],
        axis=2)

    # FG/BG means: divide by position counts  (n_eval, k, 3)
    fg_means = (fg_sums / fg_safe[None, :, None]).astype(np.float32)
    bg_means = (bg_sums / bg_safe[None, :, None]).astype(np.float32)

    # Nearest-palette lookup: Numba parallel (no large (n,p) allocation) when available;
    # fall back to mini-batched numpy otherwise.
    n_tot = n_eval * k
    if _nn_pal_idx_jit is not None:
        # One-shot: flatten (n_eval, k, 3) → (n_eval*k, 3), run numba, reshape to (n_eval, k)
        if fg_pal_norms is None:
            fg_pal_norms_f = (pal_fg * pal_fg).sum(axis=1)
        else:
            fg_pal_norms_f = np.asarray(fg_pal_norms, dtype=np.float32)
        if bg_pal_norms is None:
            bg_pal_norms_f = (pal_bg * pal_bg).sum(axis=1)
        else:
            bg_pal_norms_f = np.asarray(bg_pal_norms, dtype=np.float32)

        fg_flat = np.ascontiguousarray(fg_means.reshape(n_tot, 3))
        fg_idx = _nn_pal_idx_jit(fg_flat, pal_fg, fg_pal_norms_f)   # (n_eval*k,) int64
        fg_colors = pal_fg[fg_idx].reshape(n_eval, k, 3)             # (n_eval, k, 3)

        bg_flat = np.ascontiguousarray(bg_means.reshape(n_tot, 3))
        bg_idx = _nn_pal_idx_jit(bg_flat, pal_bg, bg_pal_norms_f)
        bg_colors = pal_bg[bg_idx].reshape(n_eval, k, 3)
    else:
        fg_colors = np.zeros((n_eval, k, 3), dtype=np.float32)
        bg_colors = np.zeros((n_eval, k, 3), dtype=np.float32)
        for b_start in range(0, k, batch_size):
            b_end = min(b_start + batch_size, k)
            bsz = b_end - b_start
            n_q = n_eval * bsz
            fg_q = np.ascontiguousarray(fg_means[:, b_start:b_end, :].reshape(n_q, 3))
            fg_colors[:, b_start:b_end, :] = (
                _nn_palette_colors(fg_q, pal_fg, fg_pal_norms).reshape(n_eval, bsz, 3))
            bg_q = np.ascontiguousarray(bg_means[:, b_start:b_end, :].reshape(n_q, 3))
            bg_colors[:, b_start:b_end, :] = (
                _nn_palette_colors(bg_q, pal_bg, bg_pal_norms).reshape(n_eval, bsz, 3))

    # Quadratic error expansion for FG:
    #   sum_{p in FG_j} ||eval_px[i,p] - fg_color[i,j]||^2
    #   = eval_sq_sum_fg[i,j]
    #     - 2 * fg_counts[j] * dot(fg_color[i,j], fg_means[i,j])
    #     + fg_counts[j] * ||fg_color[i,j]||^2
    # where fg_means[i,j] = fg_sums[i,j] / fg_counts[j]
    # so:  fg_counts[j] * dot(fg_color, fg_means)
    #     = dot(fg_color, fg_sums)  [element-wise sum over channels]
    fg_sums_d = fg_sums.astype(np.float64)
    fg_colors_d = fg_colors.astype(np.float64)
    fg_dot = (fg_colors_d * fg_sums_d).sum(axis=2)             # (n_eval, k)
    fg_color_sq = (fg_colors_d * fg_colors_d).sum(axis=2)      # (n_eval, k)
    fg_error = np.where(
        fg_counts > 0,
        eval_sq_sum_fg - 2.0 * fg_dot + fg_color_sq * fg_counts,
        0.0)

    bg_sums_d = bg_sums.astype(np.float64)
    bg_colors_d = bg_colors.astype(np.float64)
    bg_dot = (bg_colors_d * bg_sums_d).sum(axis=2)
    bg_color_sq = (bg_colors_d * bg_colors_d).sum(axis=2)
    bg_error = np.where(
        bg_counts > 0,
        eval_sq_sum_bg - 2.0 * bg_dot + bg_color_sq * bg_counts,
        0.0)

    return fg_error + bg_error   # (n_eval, k) float64


def _oklab_score_candidate(cand_pattern, member_px_oklab, member_counts, palette_oklab,
                            fg_palette_oklab=None, bg_palette_oklab=None):
    """Weighted Oklab reconstruction error for a candidate medoid pattern over cluster members.

    Args:
        cand_pattern: (8,) uint8.
        member_px_oklab: (m, 64, 3) float32 — per-position Oklab means for cluster members.
        member_counts: (m,) float32 — tile occurrence weights.
        palette_oklab: (p, 3) float32 — fallback palette.
        fg_palette_oklab: optional (p_fg, 3) float32.
        bg_palette_oklab: optional (p_bg, 3) float32.

    Returns:
        float — weighted total reconstruction cost.
    """
    cost_col = _compute_cost_column_oklab(
        cand_pattern, member_px_oklab, palette_oklab,
        fg_palette_oklab, bg_palette_oklab)
    return float((cost_col * member_counts.astype(np.float64)).sum())


def _ensure_zero_medoid(medoids):
    """Ensure medoids[0] is the all-zeros pattern (in-place).

    Flat (single-color) tiles use font_set=0, char_idx=0, so position 0 must
    always be the empty glyph.  If the all-zeros pattern already exists at
    another position it is swapped; otherwise position 0 is replaced.
    """
    zero_pattern = np.zeros(8, dtype=np.uint8)
    if np.array_equal(medoids[0], zero_pattern):
        return
    for j in range(1, len(medoids)):
        if np.array_equal(medoids[j], zero_pattern):
            medoids[j] = medoids[0].copy()
            medoids[0] = zero_pattern
            return
    medoids[0] = zero_pattern


def _collect_pattern_stats(frame_tiles, palette):
    """Scan frame tiles and accumulate per-canonical-pattern pixel statistics.

    Returns:
        patterns_arr: (n, 8) uint8 — unique canonical patterns.
        counts_arr  : (n,) float32 — tile occurrence counts.
        pixel_means : (n, 64, 3) float32 — per-position RGB means.
    """
    pattern_index = {}
    patterns = []
    counts = []
    pixel_sums = []
    for tile in frame_tiles:
        pix = tile['pixels'].astype(np.float32)
        # Fast luminance-based canonical pattern — no palette lookup needed here.
        # optimize_character_memtext's FG/BG split is driven solely by luminance
        # thresholding; the palette indices it finds are NOT used below.
        pix_flat = pix.reshape(-1, 3)  # normalise to (64, 3) regardless of input shape
        lum = 0.299 * pix_flat[:, 0] + 0.587 * pix_flat[:, 1] + 0.114 * pix_flat[:, 2]  # (64,)
        threshold = float(np.median(lum))
        fg_row = (lum >= threshold).reshape(8, 8)
        font_def = np.packbits(fg_row, axis=1).reshape(8).astype(np.uint8)
        canonical, _ = get_canonical_pattern(font_def)
        key = pattern_key(canonical)
        if key not in pattern_index:
            idx = len(patterns)
            pattern_index[key] = idx
            patterns.append(canonical)
            counts.append(0)
            pixel_sums.append(np.zeros((64, 3), dtype=np.float64))
        idx = pattern_index[key]
        counts[idx] += 1
        pixel_sums[idx] += pix_flat

    if not patterns:
        return (np.zeros((0, 8), dtype=np.uint8),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 64, 3), dtype=np.float32))

    patterns_arr = np.array(patterns, dtype=np.uint8)
    counts_arr = np.array(counts, dtype=np.float32)
    pixel_means = (np.array(pixel_sums, dtype=np.float32)
                   / counts_arr[:, None, None])
    return patterns_arr, counts_arr, pixel_means


def derive_medoids_oklab_kmeans(frame_tiles, n_medoids, palette, candidates=5,
                                fg_palette=None, bg_palette=None):
    """Derive medoids via perceptual clustering + Oklab candidate scoring.

    Key improvements over the prior implementation:

    1. **Perceptual clustering space**: KMeans is run on per-position Oklab color
       means (192-D) rather than PCA of structural byte features.  This aligns the
       clustering objective with the Oklab reconstruction-error scoring metric so
       patterns that *look* similar are grouped together.

    2. **Count-weighted KMeans**: patterns appearing more frequently pull cluster
       centers toward them, ensuring common tiles are well-covered.

    3. **Full candidate evaluation**: evaluates `candidates` centroid-nearest
       members AND the Hamming medoid of each cluster, then picks the one with
       lowest weighted Oklab reconstruction error against cluster members.
       (Prior default was candidates=1, making the scoring step a no-op.)

    4. **Split-palette aware**: accepts separate fg_palette / bg_palette so
       scoring uses the actual rendering palettes rather than a merged set.

    Args:
        frame_tiles: list of tile dicts.
        n_medoids: desired medoid count.
        palette: (N, 3) uint8 reference palette (merged when split mode is active).
        candidates: number of centroid-nearest cluster members to score per cluster.
        fg_palette: optional (N_fg, 3) uint8 FG palette for accurate scoring.
            For 4LUT mode, pass the combined (~510-entry) FG palette so scoring
            uses the full available color range.
        bg_palette: optional (N_bg, 3) uint8 BG palette for accurate scoring.
            For 4LUT mode, pass the combined (~510-entry) BG palette.
    """
    # ------------------------------------------------------------------ #
    #  Step 1: collect canonical pattern stats                             #
    # ------------------------------------------------------------------ #
    patterns_arr, counts_arr, pixel_means = _collect_pattern_stats(frame_tiles, palette)

    if len(patterns_arr) == 0:
        return np.zeros((n_medoids, 8), dtype=np.uint8)

    cluster_target = min(len(patterns_arr), n_medoids)
    if cluster_target < n_medoids:
        pad_count = n_medoids - len(patterns_arr)
        out = np.vstack([patterns_arr,
                         np.zeros((pad_count, 8), dtype=np.uint8)]) if pad_count > 0 else patterns_arr.copy()
        _ensure_zero_medoid(out)
        return out

    n_unique = len(patterns_arr)

    # ------------------------------------------------------------------ #
    #  Step 2: build perceptual + structural feature matrix                #
    # ------------------------------------------------------------------ #
    palette_oklab = rgb_to_oklab(palette.astype(np.float32))
    pixel_means_oklab = rgb_to_oklab(pixel_means.reshape(-1, 3)).reshape(n_unique, 64, 3)

    fg_palette_oklab = (rgb_to_oklab(fg_palette.astype(np.float32))
                        if fg_palette is not None else None)
    bg_palette_oklab = (rgb_to_oklab(bg_palette.astype(np.float32))
                        if bg_palette is not None else None)

    # Structural: normalised raw bytes (low weight, for tiebreaking)
    raw_bytes = patterns_arr.astype(np.float32)
    raw_n = ((raw_bytes - raw_bytes.mean(axis=0))
             / (raw_bytes.std(axis=0) + 1e-6))

    font_features = _build_perceptual_cluster_features(
        pixel_means_oklab, struct_feats=raw_n, perceptual_weight=8.0)

    # ------------------------------------------------------------------ #
    #  Step 3: count-weighted KMeans in perceptual feature space           #
    # ------------------------------------------------------------------ #
    kmeans = make_kmeans(n_clusters=cluster_target, random_state=42, n_init=10)
    try:
        kmeans.fit(font_features, sample_weight=counts_arr)
    except TypeError:
        kmeans.fit(font_features)
    labels = kmeans.labels_
    try:
        centers = kmeans.cluster_centers_
    except Exception:
        centers = np.array([
            font_features[labels == k].mean(axis=0) if np.any(labels == k) else font_features[0]
            for k in range(cluster_target)
        ])

    # ------------------------------------------------------------------ #
    #  Step 4: per-cluster candidate selection and Oklab scoring           #
    # ------------------------------------------------------------------ #
    medoids = []
    for k in range(cluster_target):
        member_idxs = np.where(labels == k)[0]
        if len(member_idxs) == 0:
            medoids.append(np.zeros(8, dtype=np.uint8))
            continue

        cand_set = set()

        # a) Top-K members nearest the cluster centroid (in feature space)
        fdists = np.linalg.norm(font_features[member_idxs] - centers[k], axis=1)
        order = np.argsort(fdists)
        n_cand = max(1, min(candidates, len(order)))
        for ri in order[:n_cand]:
            cand_set.add(int(member_idxs[ri]))

        # b) Hamming medoid of the cluster (structure-level representative)
        bitmats = np.array(
            [_pattern_bytes_to_mask(patterns_arr[gi]).astype(np.uint8) for gi in member_idxs])
        sums_k = bitmats.sum(axis=1).astype(np.int32)
        overlap_k = bitmats.dot(bitmats.T)
        ham_total = (sums_k[:, None] + sums_k[None, :] - 2 * overlap_k).sum(axis=1)
        cand_set.add(int(member_idxs[int(np.argmin(ham_total))]))

        # Score candidates: weighted Oklab error over cluster members
        member_px = pixel_means_oklab[member_idxs]   # (m, 64, 3)
        member_w = counts_arr[member_idxs]

        best_cost = float('inf')
        best_medoid = patterns_arr[member_idxs[0]]
        for ci in cand_set:
            cost = _oklab_score_candidate(
                patterns_arr[ci], member_px, member_w,
                palette_oklab, fg_palette_oklab, bg_palette_oklab)
            if cost < best_cost:
                best_cost = cost
                best_medoid = patterns_arr[ci]
        medoids.append(best_medoid)

    # ------------------------------------------------------------------ #
    #  Step 5: pad / truncate and enforce zero pattern at index 0          #
    # ------------------------------------------------------------------ #
    medoids = np.array(medoids, dtype=np.uint8)
    if len(medoids) < n_medoids:
        pad = np.zeros((n_medoids - len(medoids), 8), dtype=np.uint8)
        medoids = np.vstack([medoids, pad])
    elif len(medoids) > n_medoids:
        medoids = medoids[:n_medoids]

    _ensure_zero_medoid(medoids)
    return medoids


def _score_medoid_set(medoid_masks, pixel_means_oklab, counts_arr, palette_oklab,
                      fg_palette_oklab=None, bg_palette_oklab=None,
                      fg_pal_norms=None, bg_pal_norms=None):
    """Compute total weighted Oklab reconstruction cost for a medoid set.

    Each pattern is assigned to the medoid with minimum cost; the sum of
    weighted minimum costs is returned.

    Args:
        medoid_masks: list of (64,) bool arrays.
        pixel_means_oklab: (n, 64, 3) float32.
        counts_arr: (n,) weights.
        palette_oklab: (p, 3) float32 — fallback palette.
        fg_palette_oklab: optional (p_fg, 3) float32.
        bg_palette_oklab: optional (p_bg, 3) float32.
        fg_pal_norms: optional (p_fg,) precomputed squared norms.
        bg_pal_norms: optional (p_bg,) precomputed squared norms.
    """
    pal_fg = fg_palette_oklab if fg_palette_oklab is not None else palette_oklab
    pal_bg = bg_palette_oklab if bg_palette_oklab is not None else palette_oklab
    if fg_pal_norms is None:
        fg_pal_norms = (pal_fg * pal_fg).sum(axis=1)
    if bg_pal_norms is None:
        bg_pal_norms = (pal_bg * pal_bg).sum(axis=1)
    n_patterns = pixel_means_oklab.shape[0]
    k = len(medoid_masks)
    cost_matrix = np.full((n_patterns, k), np.inf, dtype=np.float64)

    for j, raw_mask in enumerate(medoid_masks):
        mask = np.asarray(raw_mask, dtype=bool)
        fg_pos = mask
        bg_pos = ~mask
        cost = np.zeros(n_patterns, dtype=np.float64)

        if fg_pos.sum() > 0:
            fg_means = pixel_means_oklab[:, fg_pos, :].mean(axis=1)   # (n, 3)
            chosen = _nn_palette_colors(fg_means, pal_fg, fg_pal_norms)
            dif = pixel_means_oklab[:, fg_pos, :] - chosen[:, None, :]
            cost += (dif * dif).sum(axis=(1, 2))

        if bg_pos.sum() > 0:
            bg_means = pixel_means_oklab[:, bg_pos, :].mean(axis=1)
            chosen = _nn_palette_colors(bg_means, pal_bg, bg_pal_norms)
            dif = pixel_means_oklab[:, bg_pos, :] - chosen[:, None, :]
            cost += (dif * dif).sum(axis=(1, 2))

        cost_matrix[:, j] = cost * counts_arr

    return float(cost_matrix.min(axis=1).sum())


# CLARA+PAM medoid derivation removed
# The prior `derive_medoids_clara_pam` implementation has been removed
# due to limited benefit over `oklab_kmeans` and poor performance.
# If needed in future, a replacement implementation can be reintroduced.



def compute_coherence_medoids(all_frame_canonical_sets, n_coherence):
    """Compute globally-stable coherence medoids from cross-frame pattern frequency.

    Identifies canonical patterns appearing in multiple frames and clusters them
    (weighted by frame frequency) to produce n_coherence medoids that best
    represent temporally stable patterns across the animation.

    Args:
        all_frame_canonical_sets: list of dicts, each {pattern_key: np.array} per frame
        n_coherence: number of coherence medoids to produce

    Returns:
        np.array of shape (n_coherence, 8), dtype=uint8
    """
    # Count in how many frames each pattern appears
    pattern_frame_count = {}
    pattern_data = {}
    for frame_set in all_frame_canonical_sets:
        for key, pat in frame_set.items():
            if key not in pattern_frame_count:
                pattern_frame_count[key] = 0
                pattern_data[key] = pat
            pattern_frame_count[key] += 1

    n_frames = len(all_frame_canonical_sets)
    print(f"  Coherence: {len(pattern_frame_count)} unique canonical patterns across {n_frames} frames")

    # Select patterns appearing in >= 2 frames, sorted by frequency
    multi_frame = [(key, count) for key, count in pattern_frame_count.items() if count >= 2]
    multi_frame.sort(key=lambda x: -x[1])

    print(f"  Coherence: {len(multi_frame)} patterns appear in 2+ frames")

    if not multi_frame:
        print("  Coherence: no cross-frame patterns found; returning empty set")
        return np.zeros((n_coherence, 8), dtype=np.uint8)

    patterns_list = [pattern_data[key] for key, _ in multi_frame]
    weights = np.array([count for _, count in multi_frame], dtype=np.float32)
    patterns_arr = np.array(patterns_list, dtype=np.uint8)

    if len(patterns_arr) <= n_coherence:
        result = patterns_arr.copy()
        if len(result) < n_coherence:
            pad = np.zeros((n_coherence - len(result), 8), dtype=np.uint8)
            result = np.vstack([result, pad])
        print(f"  Coherence: only {len(patterns_arr)} candidates; using all (padded to {n_coherence})")
        return result

    print(f"  Coherence: clustering {len(patterns_arr)} cross-frame patterns to {n_coherence} medoids (weighted by frame frequency)...")

    # Build feature descriptors (same pipeline as derive_medoids_from_tiles)
    n_unique = len(patterns_arr)
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

    edge_w, density_w, raw_w = _medoid_feature_weights()
    edge_feats_n = norm_feats(edge_feats) * edge_w
    density_feats_n = norm_feats(density_feats) * density_w
    raw_bytes_n = norm_feats(raw_bytes) * raw_w

    perceptual_descriptors = np.concatenate([edge_feats_n, density_feats_n, raw_bytes_n], axis=1)

    try:
        pca = PCA(n_components=min(perceptual_descriptors.shape))
        pca.fit(perceptual_descriptors)
        font_features = pca.transform(perceptual_descriptors)
    except Exception:
        font_features = perceptual_descriptors

    kmeans = make_kmeans(n_clusters=n_coherence, random_state=42, n_init=10)
    try:
        kmeans.fit(font_features, sample_weight=weights)
    except TypeError:
        # Fallback if KMeans implementation doesn't support sample_weight
        kmeans.fit(font_features)
    cluster_labels = kmeans.labels_

    coherence_medoids = compute_medoids(patterns_arr, cluster_labels, n_coherence)
    print(f"  Coherence: produced {len(coherence_medoids)} stable medoids")
    return coherence_medoids


def derive_medoids_with_coherence(frame_canonical_dict, n_medoids, coherence_medoids):
    """Derive per-frame medoids with coherence medoids reserved in the first slots.

    Coherence medoids occupy positions [0, n_coherence). Remaining slots are
    filled with frame-specific medoids from patterns not already covered by
    the coherence set.

    Args:
        frame_canonical_dict: dict {pattern_key: np.array} of frame patterns
        n_medoids: total number of medoids for this frame
        coherence_medoids: np.array (n_coherence, 8) of reserved medoids

    Returns:
        np.array of shape (n_medoids, 8)
    """
    n_coherence = len(coherence_medoids)
    n_frame_specific = n_medoids - n_coherence

    if n_frame_specific <= 0:
        return coherence_medoids[:n_medoids].copy()

    # Build set of coherence medoid keys for fast exclusion
    coherence_keys = set()
    for i in range(n_coherence):
        coherence_keys.add(pattern_key(coherence_medoids[i]))

    # Collect frame patterns not already in the coherence set
    frame_only = []
    for key, pattern in frame_canonical_dict.items():
        if key not in coherence_keys:
            frame_only.append(pattern)

    # Ensure zero tile exists somewhere
    zero_tile = np.zeros(8, dtype=np.uint8)
    zero_key = pattern_key(zero_tile)
    if zero_key not in coherence_keys and zero_key not in {pattern_key(p) for p in frame_only}:
        frame_only.insert(0, zero_tile)

    if not frame_only:
        # All frame patterns are covered by coherence set
        pad = np.zeros((n_frame_specific, 8), dtype=np.uint8)
        return np.vstack([coherence_medoids, pad])[:n_medoids]

    frame_patterns_arr = np.array(frame_only, dtype=np.uint8)
    frame_medoids = cluster_patterns_to_medoids(frame_patterns_arr, n_frame_specific)

    combined = np.vstack([coherence_medoids, frame_medoids])
    if len(combined) > n_medoids:
        combined = combined[:n_medoids]
    elif len(combined) < n_medoids:
        pad = np.zeros((n_medoids - len(combined), 8), dtype=np.uint8)
        combined = np.vstack([combined, pad])

    return combined


# ---------------------------------------------------------------------------
# Numba-accelerated high-quality tile matching kernel
# ---------------------------------------------------------------------------
def _build_numba_hq_kernel():
    """Build and return the Numba-JIT compiled _hq_assign_tiles function.

    Separated into a builder so the @njit decorator is only evaluated when
    Numba is actually available.
    """
    @numba.njit(cache=True, parallel=True)
    def _hq_assign_tiles(all_pixels, fg_palette_f, bg_palette_f, medoid_bits_f, K):
        """High-quality tile assignment — Numba-compiled, parallel over tiles.

        Parameters
        ----------
        all_pixels    : (n_tiles, 64, 3) float32  – flattened tile pixels
        fg_palette_f  : (n_fg_pal, 3) float32 – foreground palette
        bg_palette_f  : (n_bg_pal, 3) float32 – background palette
        medoid_bits_f : (n_medoids, 64) float32
        K             : int – top-K palette candidates

        Returns
        -------
        results : (n_tiles, 5) int32 – [best_medoid, fg_idx, bg_idx, invert, is_flat]
        """
        n_tiles = all_pixels.shape[0]
        n_fg_palette = fg_palette_f.shape[0]
        n_bg_palette = bg_palette_f.shape[0]
        n_medoids = medoid_bits_f.shape[0]
        results = np.empty((n_tiles, 5), dtype=np.int32)

        for i in numba.prange(n_tiles):
            orig = all_pixels[i]  # (64, 3)

            # --- Luminance --------------------------------------------------
            lum = np.empty(64, dtype=np.float32)
            for p in range(64):
                lum[p] = (0.299 * orig[p, 0]
                          + 0.587 * orig[p, 1]
                          + 0.114 * orig[p, 2])

            # Median via sort (Numba supports np.sort on 1-D)
            lum_sorted = np.sort(lum)
            threshold = 0.5 * (lum_sorted[31] + lum_sorted[32])

            # FG / BG centers
            fg_sum_r = np.float32(0.0); fg_sum_g = np.float32(0.0); fg_sum_b = np.float32(0.0)
            bg_sum_r = np.float32(0.0); bg_sum_g = np.float32(0.0); bg_sum_b = np.float32(0.0)
            fg_count = 0; bg_count = 0
            fg_mask = np.empty(64, dtype=numba.boolean)
            for p in range(64):
                if lum[p] >= threshold:
                    fg_mask[p] = True
                    fg_count += 1
                    fg_sum_r += orig[p, 0]; fg_sum_g += orig[p, 1]; fg_sum_b += orig[p, 2]
                else:
                    fg_mask[p] = False
                    bg_count += 1
                    bg_sum_r += orig[p, 0]; bg_sum_g += orig[p, 1]; bg_sum_b += orig[p, 2]

            # Check for flat (single-color) tile
            is_flat = True
            ref_r = orig[0, 0]; ref_g = orig[0, 1]; ref_b = orig[0, 2]
            for p in range(1, 64):
                if (orig[p, 0] != ref_r or orig[p, 1] != ref_g
                        or orig[p, 2] != ref_b):
                    is_flat = False
                    break
            if is_flat:
                avg_r = orig[0, 0]; avg_g = orig[0, 1]; avg_b = orig[0, 2]
                best_c = 1  # skip palette index 0
                best_d = np.float32(1e30)
                for ci in range(1, n_bg_palette):
                    dr = bg_palette_f[ci, 0] - avg_r
                    dg = bg_palette_f[ci, 1] - avg_g
                    db = bg_palette_f[ci, 2] - avg_b
                    d = dr * dr + dg * dg + db * db
                    if d < best_d:
                        best_d = d; best_c = ci
                results[i, 0] = 0
                results[i, 1] = best_c
                results[i, 2] = best_c
                results[i, 3] = 0
                results[i, 4] = 1  # flat
                continue

            fc_n = max(fg_count, 1)
            bc_n = max(bg_count, 1)
            fg_cr = fg_sum_r / fc_n; fg_cg = fg_sum_g / fc_n; fg_cb = fg_sum_b / fc_n
            bg_cr = bg_sum_r / bc_n; bg_cg = bg_sum_g / bc_n; bg_cb = bg_sum_b / bc_n

            # --- Top-K palette candidates (linear scan) --------------------
            # Maintain a small sorted list of K nearest palette indices.
            fg_top_idx = np.empty(K, dtype=np.int32)
            fg_top_d   = np.empty(K, dtype=np.float32)
            bg_top_idx = np.empty(K, dtype=np.int32)
            bg_top_d   = np.empty(K, dtype=np.float32)
            for k in range(K):
                fg_top_idx[k] = -1; fg_top_d[k] = np.float32(1e30)
                bg_top_idx[k] = -1; bg_top_d[k] = np.float32(1e30)

            for ci in range(1, n_fg_palette):  # skip index 0
                dr = fg_palette_f[ci, 0] - fg_cr
                dg = fg_palette_f[ci, 1] - fg_cg
                db = fg_palette_f[ci, 2] - fg_cb
                d = dr * dr + dg * dg + db * db
                # Insertion sort into top-K (K is tiny)
                if d < fg_top_d[K - 1]:
                    fg_top_d[K - 1] = d
                    fg_top_idx[K - 1] = ci
                    for s in range(K - 1, 0, -1):
                        if fg_top_d[s] < fg_top_d[s - 1]:
                            fg_top_d[s], fg_top_d[s - 1] = fg_top_d[s - 1], fg_top_d[s]
                            fg_top_idx[s], fg_top_idx[s - 1] = fg_top_idx[s - 1], fg_top_idx[s]
                        else:
                            break

            for ci in range(1, n_bg_palette):  # skip index 0
                dr2 = bg_palette_f[ci, 0] - bg_cr
                dg2 = bg_palette_f[ci, 1] - bg_cg
                db2 = bg_palette_f[ci, 2] - bg_cb
                d2 = dr2 * dr2 + dg2 * dg2 + db2 * db2
                if d2 < bg_top_d[K - 1]:
                    bg_top_d[K - 1] = d2
                    bg_top_idx[K - 1] = ci
                    for s in range(K - 1, 0, -1):
                        if bg_top_d[s] < bg_top_d[s - 1]:
                            bg_top_d[s], bg_top_d[s - 1] = bg_top_d[s - 1], bg_top_d[s]
                            bg_top_idx[s], bg_top_idx[s - 1] = bg_top_idx[s - 1], bg_top_idx[s]
                        else:
                            break

            # --- SSE search over K×K color pairs × n_medoids -------------
            best_sse = np.float32(1e30)
            best_m = np.int32(0)
            best_fc = fg_top_idx[0]
            best_bc = bg_top_idx[0]
            best_inv = np.int32(0)

            gain   = np.empty(64, dtype=np.float32)
            gain_i = np.empty(64, dtype=np.float32)

            for ki in range(K):
                fc = fg_top_idx[ki]
                if fc < 0:
                    continue
                fgr = fg_palette_f[fc, 0]; fgg = fg_palette_f[fc, 1]; fgb = fg_palette_f[fc, 2]
                for kj in range(K):
                    bc = bg_top_idx[kj]
                    if bc < 0:
                        continue
                    bgr = bg_palette_f[bc, 0]; bgg = bg_palette_f[bc, 1]; bgb = bg_palette_f[bc, 2]
                    # Skip if fg and bg are the same color (flat)
                    if fgr == bgr and fgg == bgg and fgb == bgb:
                        continue

                    diff_r = fgr - bgr; diff_g = fgg - bgg; diff_b = fgb - bgb
                    diff_sq = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b

                    # Normal pattern: recon_p = bg + bit_p * diff
                    base = np.float32(0.0)
                    base_i = np.float32(0.0)
                    for p in range(64):
                        er = bgr - orig[p, 0]; eg = bgg - orig[p, 1]; eb = bgb - orig[p, 2]
                        e_dot_d = er * diff_r + eg * diff_g + eb * diff_b
                        base += er * er + eg * eg + eb * eb
                        gain[p] = np.float32(2.0) * e_dot_d + diff_sq

                        eir = fgr - orig[p, 0]; eig = fgg - orig[p, 1]; eib = fgb - orig[p, 2]
                        ei_dot_d = eir * diff_r + eig * diff_g + eib * diff_b
                        base_i += eir * eir + eig * eig + eib * eib
                        gain_i[p] = np.float32(-2.0) * ei_dot_d + diff_sq

                    for m in range(n_medoids):
                        sse_n = base
                        sse_v = base_i
                        for p in range(64):
                            bit = medoid_bits_f[m, p]
                            sse_n += bit * gain[p]
                            sse_v += bit * gain_i[p]
                        if sse_n < best_sse:
                            best_sse = sse_n
                            best_m = m; best_fc = fc; best_bc = bc; best_inv = np.int32(0)
                        if sse_v < best_sse:
                            best_sse = sse_v
                            best_m = m; best_fc = fc; best_bc = bc; best_inv = np.int32(1)

            results[i, 0] = best_m
            results[i, 1] = best_fc
            results[i, 2] = best_bc
            results[i, 3] = best_inv
            results[i, 4] = 0  # not flat

        return results

    return _hq_assign_tiles


# Lazily compiled handle — filled on first use.
_numba_hq_kernel = None


def _build_numba_hq_kernel_4lut():
    """Build and return the Numba-JIT compiled _hq_assign_tiles_4lut function.

    Uses a merged-pool top-K strategy: all valid FG entries from both LUTs are
    pooled together (510 entries) and the K globally-best candidates are selected
    using perceptual (BT.601) weighted distance.  This eliminates the per-LUT K
    restriction of the previous approach, ensuring optimal candidates are found
    regardless of which LUT they belong to.  Final SSE scoring is unweighted RGB
    for consistency with compare_frames.py.

    Signature of returned function:
        _hq_assign_tiles_4lut(all_pixels, fg_merged, fg_lut_tags,
                              bg_merged, bg_lut_tags, medoid_bits_f, K)
    where
        all_pixels:   (n_tiles, 64, 3) float32
        fg_merged:    (N_fg, 3) float32  — FG0[1:] stacked with FG1[1:]
        fg_lut_tags:  (N_fg,) int32      — 0 for LUT0 entries, 1 for LUT1 entries
        bg_merged:    (N_bg, 3) float32
        bg_lut_tags:  (N_bg,) int32
        medoid_bits_f:(n_medoids, 64) float32
        K:            int — candidates to keep from each merged pool

    Returns (n_tiles, 7) int32:
        [best_medoid, fg_pal_idx, bg_pal_idx, invert, is_flat, fg_lut_bit, bg_lut_bit]
        fg_pal_idx / bg_pal_idx are 1-based indices into their respective LUT palettes.
    """
    @numba.njit(cache=True, parallel=True)
    def _hq_assign_tiles_4lut(all_pixels, fg_merged, fg_lut_tags, bg_merged, bg_lut_tags, medoid_bits_f, K):
        # --- Perceptual (BT.601) weights used in the top-K prefilter distance ---
        WR = np.float32(0.299)
        WG = np.float32(0.587)
        WB = np.float32(0.114)

        n_tiles       = all_pixels.shape[0]
        n_fg_merged   = fg_merged.shape[0]
        n_bg_merged   = bg_merged.shape[0]
        n_medoids     = medoid_bits_f.shape[0]
        results = np.empty((n_tiles, 7), dtype=np.int32)

        for i in numba.prange(n_tiles):
            orig = all_pixels[i]   # (64, 3)

            # ---- luminance split (BT.601, consistent with prefilter) -------
            lum = np.empty(64, dtype=np.float32)
            for p in range(64):
                lum[p] = WR * orig[p, 0] + WG * orig[p, 1] + WB * orig[p, 2]
            lum_sorted = np.sort(lum)
            threshold = np.float32(0.5) * (lum_sorted[31] + lum_sorted[32])

            fg_sum_r = np.float32(0.); fg_sum_g = np.float32(0.); fg_sum_b = np.float32(0.)
            bg_sum_r = np.float32(0.); bg_sum_g = np.float32(0.); bg_sum_b = np.float32(0.)
            fg_count = np.int32(0); bg_count = np.int32(0)

            is_flat = True
            ref_r = orig[0, 0]; ref_g = orig[0, 1]; ref_b = orig[0, 2]
            for p in range(64):
                if p > 0 and (orig[p, 0] != ref_r or orig[p, 1] != ref_g or orig[p, 2] != ref_b):
                    is_flat = False
                if lum[p] >= threshold:
                    fg_count += 1
                    fg_sum_r += orig[p, 0]; fg_sum_g += orig[p, 1]; fg_sum_b += orig[p, 2]
                else:
                    bg_count += 1
                    bg_sum_r += orig[p, 0]; bg_sum_g += orig[p, 1]; bg_sum_b += orig[p, 2]

            if is_flat:
                avg_r = orig[0, 0]; avg_g = orig[0, 1]; avg_b = orig[0, 2]
                best_c = np.int32(0); best_d = np.float32(1e30); best_lut = np.int32(0)
                # search merged BG pool (perceptual distance)
                for ci in range(n_bg_merged):
                    dr = bg_merged[ci, 0] - avg_r
                    dg = bg_merged[ci, 1] - avg_g
                    db = bg_merged[ci, 2] - avg_b
                    d = WR*dr*dr + WG*dg*dg + WB*db*db
                    if d < best_d:
                        best_d = d; best_c = ci; best_lut = bg_lut_tags[ci]
                # Map merged index → 1-based palette index
                # Layout: merged[0..254]=LUT0[1..255], merged[255..509]=LUT1[1..255]
                if best_lut == 0:
                    pal_idx = best_c + 1
                else:
                    pal_idx = best_c - 254
                results[i, 0] = 0; results[i, 1] = pal_idx; results[i, 2] = pal_idx
                results[i, 3] = 0; results[i, 4] = 1; results[i, 5] = best_lut; results[i, 6] = best_lut
                continue

            fc_n = max(fg_count, np.int32(1))
            bc_n = max(bg_count, np.int32(1))
            fg_cr = fg_sum_r / fc_n; fg_cg = fg_sum_g / fc_n; fg_cb = fg_sum_b / fc_n
            bg_cr = bg_sum_r / bc_n; bg_cg = bg_sum_g / bc_n; bg_cb = bg_sum_b / bc_n

            # ---- Top-K from merged FG pool (perceptual distance) -----
            fg_top_idx = np.empty(K, dtype=np.int32)
            fg_top_d   = np.empty(K, dtype=np.float32)
            for k in range(K):
                fg_top_idx[k] = np.int32(-1); fg_top_d[k] = np.float32(1e30)

            for ci in range(n_fg_merged):
                dr = fg_merged[ci, 0] - fg_cr
                dg = fg_merged[ci, 1] - fg_cg
                db = fg_merged[ci, 2] - fg_cb
                d = WR*dr*dr + WG*dg*dg + WB*db*db
                if d < fg_top_d[K - 1]:
                    fg_top_d[K - 1] = d; fg_top_idx[K - 1] = ci
                    for s in range(K - 1, 0, -1):
                        if fg_top_d[s] < fg_top_d[s - 1]:
                            fg_top_d[s], fg_top_d[s - 1] = fg_top_d[s - 1], fg_top_d[s]
                            fg_top_idx[s], fg_top_idx[s - 1] = fg_top_idx[s - 1], fg_top_idx[s]
                        else:
                            break

            # ---- Top-K from merged BG pool (perceptual distance) -----
            bg_top_idx = np.empty(K, dtype=np.int32)
            bg_top_d   = np.empty(K, dtype=np.float32)
            for k in range(K):
                bg_top_idx[k] = np.int32(-1); bg_top_d[k] = np.float32(1e30)

            for ci in range(n_bg_merged):
                dr = bg_merged[ci, 0] - bg_cr
                dg = bg_merged[ci, 1] - bg_cg
                db = bg_merged[ci, 2] - bg_cb
                d = WR*dr*dr + WG*dg*dg + WB*db*db
                if d < bg_top_d[K - 1]:
                    bg_top_d[K - 1] = d; bg_top_idx[K - 1] = ci
                    for s in range(K - 1, 0, -1):
                        if bg_top_d[s] < bg_top_d[s - 1]:
                            bg_top_d[s], bg_top_d[s - 1] = bg_top_d[s - 1], bg_top_d[s]
                            bg_top_idx[s], bg_top_idx[s - 1] = bg_top_idx[s - 1], bg_top_idx[s]
                        else:
                            break

            # ---- SSE search over K×K pairs × n_medoids (unweighted for consistency) ----
            best_sse  = np.float32(1e30)
            best_m    = np.int32(0)
            best_fc   = np.int32(1); best_fl = np.int32(0)
            best_bc   = np.int32(1); best_bl = np.int32(0)
            best_inv  = np.int32(0)

            gain   = np.empty(64, dtype=np.float32)
            gain_i = np.empty(64, dtype=np.float32)

            for ki in range(K):
                fci = fg_top_idx[ki]
                if fci < 0:
                    continue
                fgr = fg_merged[fci, 0]; fgg = fg_merged[fci, 1]; fgb = fg_merged[fci, 2]
                fl = fg_lut_tags[fci]
                if fl == 0:
                    fc_pal_idx = fci + 1
                else:
                    fc_pal_idx = fci - 254  # merged[255..509] → pal[1..255]

                for kj in range(K):
                    bci = bg_top_idx[kj]
                    if bci < 0:
                        continue
                    bgr = bg_merged[bci, 0]; bgg = bg_merged[bci, 1]; bgb = bg_merged[bci, 2]
                    bl = bg_lut_tags[bci]
                    if bl == 0:
                        bc_pal_idx = bci + 1
                    else:
                        bc_pal_idx = bci - 254

                    if fgr == bgr and fgg == bgg and fgb == bgb:
                        continue

                    diff_r = fgr - bgr; diff_g = fgg - bgg; diff_b = fgb - bgb
                    diff_sq = diff_r*diff_r + diff_g*diff_g + diff_b*diff_b

                    base   = np.float32(0.)
                    base_i = np.float32(0.)
                    for p in range(64):
                        er = bgr - orig[p, 0]; eg = bgg - orig[p, 1]; eb = bgb - orig[p, 2]
                        e_dot_d = er*diff_r + eg*diff_g + eb*diff_b
                        base += er*er + eg*eg + eb*eb
                        gain[p] = np.float32(2.) * e_dot_d + diff_sq

                        eir = fgr - orig[p, 0]; eig = fgg - orig[p, 1]; eib = fgb - orig[p, 2]
                        ei_dot_d = eir*diff_r + eig*diff_g + eib*diff_b
                        base_i += eir*eir + eig*eig + eib*eib
                        gain_i[p] = np.float32(-2.) * ei_dot_d + diff_sq

                    for m in range(n_medoids):
                        sse_n = base; sse_v = base_i
                        for p in range(64):
                            bit = medoid_bits_f[m, p]
                            sse_n += bit * gain[p]
                            sse_v += bit * gain_i[p]
                        if sse_n < best_sse:
                            best_sse = sse_n; best_m = m
                            best_fc = fc_pal_idx; best_bc = bc_pal_idx
                            best_inv = np.int32(0); best_fl = fl; best_bl = bl
                        if sse_v < best_sse:
                            best_sse = sse_v; best_m = m
                            best_fc = fc_pal_idx; best_bc = bc_pal_idx
                            best_inv = np.int32(1); best_fl = fl; best_bl = bl

            results[i, 0] = best_m
            results[i, 1] = best_fc
            results[i, 2] = best_bc
            results[i, 3] = best_inv
            results[i, 4] = 0  # not flat
            results[i, 5] = best_fl
            results[i, 6] = best_bl

        return results

    return _hq_assign_tiles_4lut


# Lazily compiled handles — filled on first use.
_numba_hq_kernel_4lut = None


def assign_tiles_to_medoids(frame_tiles, palette, clustered_medoids, high_quality=False, bg_palette=None,
                            four_color_luts=False, fg_lut_palettes=None, bg_lut_palettes=None):
    """Assign each tile to best medoid/font-set and FG/BG indices.

    Default: Hamming-distance matching with luminance-based FG/BG split.
    When high_quality=True, evaluates top-K palette-color candidates and
    selects the (FG, BG, medoid, invert) combination that minimises actual
    pixel-level reconstruction error.

    If Numba is available the high-quality path is JIT-compiled and
    parallelised across CPU cores for a significant speed-up.

    Args:
        frame_tiles: list of tile dicts with 'pixels' key (8x8x3 uint8)
        palette: (256, 3) uint8 RGB palette (used as FG palette)
        clustered_medoids: (n_medoids, 8) uint8 font patterns
        high_quality: if True, use reconstruction-error search (slower, better)
        bg_palette: (256, 3) uint8 RGB palette for background.
                    If None, uses palette for both FG and BG (shared mode).
    """
    global _numba_hq_kernel, _numba_hq_kernel_4lut

    if bg_palette is None:
        bg_palette = palette

    n_medoids = clustered_medoids.shape[0]
    n_tiles = len(frame_tiles)
    n_font_sets = max(1, n_medoids // 256)

    if four_color_luts:
        tile_assignments = np.zeros((n_tiles, 7), dtype=np.uint16)
    else:
        tile_assignments = np.zeros((n_tiles, 5), dtype=np.uint16)

    # Precompute medoid bit patterns: (n_medoids, 64)
    medoid_bits = np.zeros((n_medoids, 64), dtype=np.uint8)
    for m_idx in range(n_medoids):
        pattern = clustered_medoids[m_idx]
        for r in range(8):
            byte = int(pattern[r])
            for c in range(8):
                medoid_bits[m_idx, r * 8 + c] = (byte >> (7 - c)) & 1

    # Prepare palette float buffers. When four_color_luts is enabled we expect
    # fg_lut_palettes and bg_lut_palettes to be lists of two (256,3) arrays.
    if four_color_luts:
        if fg_lut_palettes is None or bg_lut_palettes is None:
            raise ValueError('four_color_luts=True but fg_lut_palettes/bg_lut_palettes not provided')
        fg0_f = np.array(fg_lut_palettes[0], dtype=np.float32)
        fg1_f = np.array(fg_lut_palettes[1], dtype=np.float32)
        bg0_f = np.array(bg_lut_palettes[0], dtype=np.float32)
        bg1_f = np.array(bg_lut_palettes[1], dtype=np.float32)
        # Build merged pools for the Numba 4-LUT kernel (skip index 0 = transparent).
        # Layout: merged[0..254] = LUT0[1..255], merged[255..509] = LUT1[1..255].
        fg_merged = np.vstack([fg0_f[1:], fg1_f[1:]])     # (510, 3) float32
        fg_lut_tags = np.zeros(510, dtype=np.int32)
        fg_lut_tags[255:] = 1
        bg_merged = np.vstack([bg0_f[1:], bg1_f[1:]])     # (510, 3) float32
        bg_lut_tags = np.zeros(510, dtype=np.int32)
        bg_lut_tags[255:] = 1
    else:
        fg_palette_f = palette.astype(np.float32)
        bg_palette_f = bg_palette.astype(np.float32)
    medoid_bits_f = medoid_bits.astype(np.float32)  # (n_medoids, 64)

    K = 3       # top-K palette candidates for the 2-LUT high-quality kernel
    K_4lut = 6  # top-K for 4-LUT merged-pool search (6×6=36 pairs, matching 2-LUT compute budget)

    # ----- Numba fast-path for high_quality --------------------------------
    # NOTE: numba kernel does not support multi-LUT (four_color_luts) mode.
    if high_quality and _USE_NUMBA and not four_color_luts:
        # Pack all tile pixels into a contiguous (n_tiles, 64, 3) float32 array
        all_pixels = np.empty((n_tiles, 64, 3), dtype=np.float32)
        for i, tile in enumerate(frame_tiles):
            all_pixels[i] = tile['pixels'].reshape(64, 3).astype(np.float32)

        # Lazily compile the kernel on first call
        if _numba_hq_kernel is None:
            print("  [numba] JIT-compiling high-quality kernel (first call only)...")
            _numba_hq_kernel = _build_numba_hq_kernel()

        hq_results = _numba_hq_kernel(all_pixels, fg_palette_f, bg_palette_f, medoid_bits_f, K)
        # hq_results: (n_tiles, 5) int32 — [best_medoid, fg_idx, bg_idx, invert, is_flat]

        for i in range(n_tiles):
            best_medoid = int(hq_results[i, 0])
            fg_idx      = int(hq_results[i, 1])
            bg_idx      = int(hq_results[i, 2])
            best_invert = int(hq_results[i, 3])
            is_flat     = int(hq_results[i, 4])

            # Flat tile: all pixels identical → font-set 0, char 0
            if is_flat:
                tile_assignments[i] = [0, 0, fg_idx, bg_idx, 0]
            else:
                fs = best_medoid // 256
                if fs >= n_font_sets:
                    fs = n_font_sets - 1
                char_idx = best_medoid % 256
                tile_assignments[i] = [fs, char_idx, fg_idx, bg_idx, best_invert]

        return tile_assignments

    # ----- Numba fast-path for high_quality + four_color_luts ---------------
    if high_quality and _USE_NUMBA and four_color_luts:
        all_pixels = np.empty((n_tiles, 64, 3), dtype=np.float32)
        for i, tile in enumerate(frame_tiles):
            all_pixels[i] = tile['pixels'].reshape(64, 3).astype(np.float32)

        if _numba_hq_kernel_4lut is None:
            print("  [numba] JIT-compiling 4-LUT high-quality kernel (first call only)...")
            _numba_hq_kernel_4lut = _build_numba_hq_kernel_4lut()

        # hq4_results: (n_tiles, 7) int32
        # cols: [best_medoid, fg_idx, bg_idx, invert, is_flat, fg_lut_bit, bg_lut_bit]
        hq4_results = _numba_hq_kernel_4lut(all_pixels, fg_merged, fg_lut_tags, bg_merged, bg_lut_tags, medoid_bits_f, K_4lut)

        for i in range(n_tiles):
            best_medoid = int(hq4_results[i, 0])
            fg_idx      = int(hq4_results[i, 1])
            bg_idx      = int(hq4_results[i, 2])
            best_invert = int(hq4_results[i, 3])
            is_flat     = int(hq4_results[i, 4])
            fg_lut_bit  = int(hq4_results[i, 5])
            bg_lut_bit  = int(hq4_results[i, 6])

            if is_flat:
                tile_assignments[i] = [0, 0, fg_idx, bg_idx, 0, fg_lut_bit, bg_lut_bit]
            else:
                fs = best_medoid // 256
                if fs >= n_font_sets:
                    fs = n_font_sets - 1
                char_idx = best_medoid % 256
                tile_assignments[i] = [fs, char_idx, fg_idx, bg_idx, best_invert, fg_lut_bit, bg_lut_bit]

        return tile_assignments

    # ----- Scalar (non-Numba) path -----------------------------------------
    for i, tile in enumerate(frame_tiles):
        orig_rgb = tile['pixels'].reshape(-1, 3).astype(np.float32)  # (64, 3)

        unique_colors = np.unique(orig_rgb, axis=0)
        if len(unique_colors) < 2:
            avg = orig_rgb.mean(axis=0)
            if four_color_luts:
                d0 = np.linalg.norm(bg0_f - avg, axis=1)
                d1 = np.linalg.norm(bg1_f - avg, axis=1)
                d0[0] = float('inf')
                d1[0] = float('inf')
                i0 = int(np.argmin(d0))
                i1 = int(np.argmin(d1))
                if d0[i0] <= d1[i1]:
                    best_color = i0
                    best_bg_lut = 0
                else:
                    best_color = i1
                    best_bg_lut = 1
                # For flat tiles choose same LUT for FG/BG
                tile_assignments[i] = [0, 0, best_color, best_color, 0, best_bg_lut, best_bg_lut]
            else:
                dists = np.linalg.norm(bg_palette_f - avg, axis=1)
                dists[0] = float('inf')
                best_color = int(np.argmin(dists))
                tile_assignments[i] = [0, 0, best_color, best_color, 0]
            continue

        # Luminance-based FG/BG split
        luminance = 0.299 * orig_rgb[:, 0] + 0.587 * orig_rgb[:, 1] + 0.114 * orig_rgb[:, 2]
        threshold = np.median(luminance)
        fg_mask = luminance >= threshold
        bg_mask = ~fg_mask

        fg_center = orig_rgb[fg_mask].mean(axis=0) if np.any(fg_mask) else orig_rgb.mean(axis=0)
        bg_center = orig_rgb[bg_mask].mean(axis=0) if np.any(bg_mask) else orig_rgb.mean(axis=0)

        if high_quality:
            # --- Reconstruction-error search over top-K palette candidates ---
            # Uses an algebraic expansion of per-pixel SSE so that the inner
            # loop is a single (n_medoids, 64) @ (64,) matrix-vector product
            # per (fg, bg, inversion) candidate — very fast.
            # For four-color-luts evaluate FG0/1 x BG0/1 combinations; otherwise
            # run the single-palette search as before.
            best_sse = float('inf')
            best_result = (0, 0, 0, 0, 0, 0)

            def eval_candidates(fg_pal_f, bg_pal_f, fg_lut_bit, bg_lut_bit):
                nonlocal best_sse, best_result
                fg_dists_all = np.linalg.norm(fg_pal_f - fg_center, axis=1)
                fg_dists_all[0] = float('inf')
                fg_candidates = np.argpartition(fg_dists_all, K)[:K]

                bg_dists_all = np.linalg.norm(bg_pal_f - bg_center, axis=1)
                bg_dists_all[0] = float('inf')
                bg_candidates = np.argpartition(bg_dists_all, K)[:K]

                for fc in fg_candidates:
                    fg_color = fg_pal_f[fc]
                    for bc in bg_candidates:
                        bg_color = bg_pal_f[bc]
                        if np.allclose(fg_color, bg_color):
                            continue
                        diff = fg_color - bg_color
                        diff_sq = float(np.dot(diff, diff))

                        e = bg_color - orig_rgb
                        e_dot_d = (e * diff).sum(axis=1)
                        base = float((e * e).sum())
                        gain = 2.0 * e_dot_d + diff_sq
                        sse_normal = base + medoid_bits_f @ gain
                        mn = int(np.argmin(sse_normal))
                        if sse_normal[mn] < best_sse:
                            best_sse = float(sse_normal[mn])
                            best_result = (mn, int(fc), int(bc), 0, fg_lut_bit, bg_lut_bit)

                        ei = fg_color - orig_rgb
                        ei_dot_d = (ei * diff).sum(axis=1)
                        base_i = float((ei * ei).sum())
                        gain_i = -2.0 * ei_dot_d + diff_sq
                        sse_inv = base_i + medoid_bits_f @ gain_i
                        mi = int(np.argmin(sse_inv))
                        if sse_inv[mi] < best_sse:
                            best_sse = float(sse_inv[mi])
                            best_result = (mi, int(fc), int(bc), 1, fg_lut_bit, bg_lut_bit)

            if four_color_luts:
                eval_candidates(fg0_f, bg0_f, 0, 0)
                eval_candidates(fg0_f, bg1_f, 0, 1)
                eval_candidates(fg1_f, bg0_f, 1, 0)
                eval_candidates(fg1_f, bg1_f, 1, 1)
            else:
                eval_candidates(fg_palette_f, bg_palette_f, 0, 0)

            best_medoid, fg_idx, bg_idx, best_invert, best_fg_lut, best_bg_lut = best_result

        else:
            # --- Legacy Hamming-distance matching ---
            if four_color_luts:
                # Find best FG across both FG palettes
                fg_d0 = np.linalg.norm(fg0_f - fg_center, axis=1)
                fg_d0[0] = float('inf')
                fg_d1 = np.linalg.norm(fg1_f - fg_center, axis=1)
                fg_d1[0] = float('inf')
                fg_idx0 = int(np.argmin(fg_d0))
                fg_idx1 = int(np.argmin(fg_d1))
                if fg_d0[fg_idx0] <= fg_d1[fg_idx1]:
                    fg_idx = fg_idx0
                    fg_lut_bit = 0
                else:
                    fg_idx = fg_idx1
                    fg_lut_bit = 1

                # Best BG across both BG palettes
                bg_d0 = np.linalg.norm(bg0_f - bg_center, axis=1)
                bg_d0[0] = float('inf')
                bg_d1 = np.linalg.norm(bg1_f - bg_center, axis=1)
                bg_d1[0] = float('inf')
                bg_idx0 = int(np.argmin(bg_d0))
                bg_idx1 = int(np.argmin(bg_d1))
                if bg_d0[bg_idx0] <= bg_d1[bg_idx1]:
                    bg_idx = bg_idx0
                    bg_lut_bit = 0
                else:
                    bg_idx = bg_idx1
                    bg_lut_bit = 1

                # If chosen colors are numerically identical, prefer other BG candidate
                fg_color = (fg0_f if fg_lut_bit == 0 else fg1_f)[fg_idx]
                bg_color = (bg0_f if bg_lut_bit == 0 else bg1_f)[bg_idx]
                if np.allclose(fg_color, bg_color):
                    # try to pick alternative bg
                    if bg_lut_bit == 0 and bg_d1[bg_idx1] < float('inf'):
                        bg_idx = bg_idx1
                        bg_lut_bit = 1
                    elif bg_lut_bit == 1 and bg_d0[bg_idx0] < float('inf'):
                        bg_idx = bg_idx0
                        bg_lut_bit = 0

                # Medoid selection (pattern only)
                ideal_pattern = np.zeros(8, dtype=np.uint8)
                for row in range(8):
                    byte = 0
                    for col in range(8):
                        pixel_idx = row * 8 + col
                        if fg_mask[pixel_idx]:
                            byte |= (1 << (7 - col))
                    ideal_pattern[row] = byte

                ideal_bits = np.zeros(64, dtype=np.uint8)
                for r in range(8):
                    byte = ideal_pattern[r]
                    for c in range(8):
                        ideal_bits[r * 8 + c] = (byte >> (7 - c)) & 1

                xor_normal = np.bitwise_xor(medoid_bits, ideal_bits)
                hamming_normal = np.sum(xor_normal.astype(np.float32), axis=1)

                ideal_bits_inv = 1 - ideal_bits
                xor_inv = np.bitwise_xor(medoid_bits, ideal_bits_inv)
                hamming_inv = np.sum(xor_inv.astype(np.float32), axis=1)

                best_normal_idx = int(np.argmin(hamming_normal))
                best_inv_idx = int(np.argmin(hamming_inv))

                if hamming_normal[best_normal_idx] <= hamming_inv[best_inv_idx]:
                    best_medoid = best_normal_idx
                    best_invert = 0
                else:
                    best_medoid = best_inv_idx
                    best_invert = 1

                fs = best_medoid // 256
                if fs >= n_font_sets:
                    fs = n_font_sets - 1
                char_idx = best_medoid % 256
                tile_assignments[i] = [fs, char_idx, fg_idx, bg_idx, best_invert, fg_lut_bit, bg_lut_bit]
                continue
            else:
                # Legacy single-palette Hamming match (no four-color LUTs)
                fg_d = np.linalg.norm(fg_palette_f - fg_center, axis=1)
                fg_d[0] = float('inf')
                fg_idx = int(np.argmin(fg_d))

                bg_d = np.linalg.norm(bg_palette_f - bg_center, axis=1)
                bg_d[0] = float('inf')
                bg_idx = int(np.argmin(bg_d))

                # Ensure distinct FG/BG
                if fg_idx == bg_idx:
                    bg_d[bg_idx] = float('inf')
                    bg_idx = int(np.argmin(bg_d))

                # Determine ideal pattern and choose best medoid (normal/inv)
                ideal_pattern = np.zeros(8, dtype=np.uint8)
                for row in range(8):
                    byte = 0
                    for col in range(8):
                        pixel_idx = row * 8 + col
                        if fg_mask[pixel_idx]:
                            byte |= (1 << (7 - col))
                    ideal_pattern[row] = byte

                ideal_bits = np.zeros(64, dtype=np.uint8)
                for r in range(8):
                    byte = ideal_pattern[r]
                    for c in range(8):
                        ideal_bits[r * 8 + c] = (byte >> (7 - c)) & 1

                xor_normal = np.bitwise_xor(medoid_bits, ideal_bits)
                hamming_normal = np.sum(xor_normal.astype(np.float32), axis=1)

                ideal_bits_inv = 1 - ideal_bits
                xor_inv = np.bitwise_xor(medoid_bits, ideal_bits_inv)
                hamming_inv = np.sum(xor_inv.astype(np.float32), axis=1)

                best_normal_idx = int(np.argmin(hamming_normal))
                best_inv_idx = int(np.argmin(hamming_inv))

                if hamming_normal[best_normal_idx] <= hamming_inv[best_inv_idx]:
                    best_medoid = best_normal_idx
                    best_invert = 0
                else:
                    best_medoid = best_inv_idx
                    best_invert = 1

                fs = best_medoid // 256
                if fs >= n_font_sets:
                    fs = n_font_sets - 1
                char_idx = best_medoid % 256
                tile_assignments[i] = [fs, char_idx, fg_idx, bg_idx, best_invert]
                continue
        # End scalar loop

        fs = best_medoid // 256
        if fs >= n_font_sets:
            fs = n_font_sets - 1
        char_idx = best_medoid % 256

        if four_color_luts:
            tile_assignments[i] = [fs, char_idx, int(fg_idx), int(bg_idx), int(best_invert), int(best_fg_lut), int(best_bg_lut)]
        else:
            tile_assignments[i] = [fs, char_idx, int(fg_idx), int(bg_idx), int(best_invert)]

    return tile_assignments


def process_animation_frames_memtext_global(input_dir, high_quality=False, split_palette=False, denoise=0,
                                             medoids_arr=None, four_color_luts=False):
    """Process animation frames in MEMTEXT mode.
    
    Key differences from regular animation mode:
    - 256-color FG/BG palettes (identical, split with --split-palette, or 4 sets of 256 for 4LUT mode)
    - 1024 medoids in 4 font sets of 256 each
    - Pattern inversion support to reduce unique patterns
    - Tile data includes font_set (0-3), char_index (0-255), lut_id, and invert bit
    
    Args:
        input_dir: Directory containing input frames
        high_quality: if True, use reconstruction-error tile matching
        split_palette: if True, derive separate FG/BG palettes (512 total colors)
        denoise: max island size to remove from medoids (0=disabled)
        medoids_arr: optional np.array (1024, 8) uint8 — precomputed medoids loaded
            from a .npy file.  When provided, the per-animation clustering step is
            skipped and these medoids are used directly (after optional denoising).
        four_color_luts: if True, create 4 separate palettes and evaluate all 4 LUT combinations.
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
    
    # Step 1: Create palette(s)
    fg_palettes = None
    bg_palettes = None
    if four_color_luts:
        fg_palettes, bg_palettes = create_memtext_4lut_palettes(image_files, split_palette=split_palette)
        # Use first FG palette as reference for medoid derivation unless
        # MEDOID_METHOD requests a merged/sample-aware palette.
        if MEDOID_METHOD != 'legacy':
            palette = merge_palettes_for_medoids(fg_palettes, bg_palettes)
        else:
            palette = fg_palettes[0]
        fg_palette = fg_palettes[0]
        bg_palette = bg_palettes[0]
    else:
        if split_palette:
            fg_palette, bg_palette = create_memtext_512_color_palette(image_files)
            if MEDOID_METHOD != 'legacy':
                palette = merge_palettes_for_medoids([fg_palette], [bg_palette])
            else:
                palette = fg_palette  # used as reference for medoid derivation
        else:
            palette = create_memtext_256_color_palette(image_files)
            fg_palette = bg_palette = palette
    
    # Step 2: Load all frames and extract tiles
    print("Extracting 8x8 patterns from all frames...")
    all_tiles = []
    frame_tile_indices = []

    for idx, image_file in enumerate(image_files):
        print(f"Loading frame {idx+1}/{len(image_files)}: {os.path.basename(image_file)}")
        image = load_and_scale_image(image_file)
        frame_tiles = extract_frame_tiles(image)
        start_idx = len(all_tiles)
        all_tiles.extend(frame_tiles)
        end_idx = len(all_tiles)
        frame_tile_indices.append((start_idx, end_idx))
    
    print(f"Extracted {len(all_tiles)} total tiles")
    
    # Step 3: Derive 1024 medoids (cluster from tiles, or use precomputed set)
    if medoids_arr is not None:
        print(f"Using precomputed medoids ({len(medoids_arr)} patterns) — skipping per-animation clustering")
        clustered_medoids = medoids_arr.copy()
        if denoise > 0:
            clustered_medoids = denoise_medoid_set(clustered_medoids, denoise)
    else:
        print("Clustering to 1024 medoids...")
        # For 4LUT + enhanced methods, score against all 4 palette LUTs combined
        # (~510 FG colors and ~510 BG colors) so medoid selection sees the full
        # 1024-color space rather than only the first 256-entry LUT.
        if four_color_luts and fg_palettes is not None and MEDOID_METHOD != 'legacy':
            _med_fg = np.concatenate([fg_palettes[0][1:], fg_palettes[1][1:]], axis=0)
            _med_bg = np.concatenate([bg_palettes[0][1:], bg_palettes[1][1:]], axis=0)
            print(f"  4LUT medoid scoring: {len(_med_fg)} FG + {len(_med_bg)} BG palette entries")
        else:
            _med_fg = fg_palette
            _med_bg = bg_palette
        clustered_medoids = derive_medoids_from_tiles(all_tiles, 1024, palette,
            fg_palette=_med_fg, bg_palette=_med_bg)
        if denoise > 0:
            clustered_medoids = denoise_medoid_set(clustered_medoids, denoise)
    print(f"Using {len(clustered_medoids)} medoids")
    
    # Split into 4 font sets of 256 each
    font_sets = [clustered_medoids[i*256:(i+1)*256] for i in range(4)]
    
    print(f"Split medoids into 4 font sets of 256 patterns each")
    
    # Step 4: Process each frame - assign tiles to medoids + colors
    print("Assigning tiles to medoids for each frame...")
    frames_data = []

    for frame_idx, (start, end) in enumerate(frame_tile_indices):
        print(f"Processing frame {frame_idx+1}/{len(frame_tile_indices)}...")
        frame_tiles = all_tiles[start:end]
        tile_assignments = assign_tiles_to_medoids(
            frame_tiles, fg_palette, clustered_medoids, high_quality=high_quality,
            bg_palette=bg_palette, four_color_luts=four_color_luts,
            fg_lut_palettes=fg_palettes, bg_lut_palettes=bg_palettes)
        
        frames_data.append({
            'width': 640,
            'height': 480,
            'use_single_row_blocks': False,
            'memtext': True,
            'tile_assignments': tile_assignments,  # (n_tiles, 5): fs, char, fg, bg, invert
            'palette': fg_palette,
            'bg_palette': bg_palette,
            'font_sets': font_sets,
        })
    
    # Store shared data in first frame for output
    frames_data[0]['global_palette'] = fg_palette
    frames_data[0]['global_bg_palette'] = bg_palette
    frames_data[0]['global_font_sets'] = font_sets
    if four_color_luts and fg_palettes is not None and bg_palettes is not None:
        frames_data[0]['lut_palettes'] = {'fg': fg_palettes, 'bg': bg_palettes}
    
    return frames_data


def process_animation_frames_memtext_hybrid(input_dir, n_coherence=0, high_quality=False, split_palette=False, denoise=0, four_color_luts=False):
    """Process animation frames in MEMTEXT hybrid mode.

    Hybrid mode characteristics (workaround for FPGA LUT 1 bug):
    - Global palette(s) derived from all frames (output once, chunk_id==lut_id)
    - Per-frame 512 medoids split into 2 font sets of 256 (output per frame)
    - Font set IDs alternate between pairs (0,1) and (2,3) for double buffering
    - No per-frame color palette output
    - Optional coherence medoids for cross-frame temporal stability
    """
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

    # Step 1: Create global palette(s)
    fg_palettes = None
    bg_palettes = None
    if four_color_luts:
        fg_palettes, bg_palettes = create_memtext_4lut_palettes(image_files, split_palette=split_palette)
        if MEDOID_METHOD != 'legacy':
            palette = merge_palettes_for_medoids(fg_palettes, bg_palettes)
        else:
            palette = fg_palettes[0]
        fg_palette = fg_palettes[0]
        bg_palette = bg_palettes[0]
    else:
        if split_palette:
            fg_palette, bg_palette = create_memtext_512_color_palette(image_files)
            if MEDOID_METHOD != 'legacy':
                palette = merge_palettes_for_medoids([fg_palette], [bg_palette])
            else:
                palette = fg_palette
        else:
            palette = create_memtext_256_color_palette(image_files)
            fg_palette = bg_palette = palette

    frames_data = []

    if n_coherence > 0:
        # Two-pass: scan all frames for canonical patterns, compute coherence, then process
        print("Pass 1: Extracting canonical patterns from all frames...")
        all_frame_tiles = []
        all_frame_canonical_sets = []
        for frame_idx, image_file in enumerate(image_files):
            print(f"  Pre-scan frame {frame_idx + 1}/{len(image_files)}: {os.path.basename(image_file)}")
            image = load_and_scale_image(image_file)
            frame_tiles = extract_frame_tiles(image)
            all_frame_tiles.append(frame_tiles)
            canonical_set = extract_canonical_patterns(frame_tiles, palette)
            all_frame_canonical_sets.append(canonical_set)

        coherence_medoids = compute_coherence_medoids(all_frame_canonical_sets, n_coherence)

        print("Pass 2: Per-frame medoid derivation with coherence...")
        for frame_idx in range(len(image_files)):
            print(f"  Processing frame {frame_idx + 1}/{len(image_files)}...")
            frame_tiles = all_frame_tiles[frame_idx]
            frame_canonical = all_frame_canonical_sets[frame_idx]

            combined_medoids = derive_medoids_with_coherence(
                frame_canonical, 512, coherence_medoids)
            if denoise > 0:
                combined_medoids = denoise_medoid_set(combined_medoids, denoise)
            font_sets_local = [combined_medoids[i * 256:(i + 1) * 256] for i in range(2)]

            tile_assignments = assign_tiles_to_medoids(
                frame_tiles, fg_palette, combined_medoids, high_quality=high_quality,
                bg_palette=bg_palette, four_color_luts=four_color_luts,
                fg_lut_palettes=fg_palettes, bg_lut_palettes=bg_palettes)

            frames_data.append({
                'width': 640,
                'height': 480,
                'use_single_row_blocks': False,
                'memtext': True,
                'tile_assignments': tile_assignments,
                'palette': fg_palette,
                'bg_palette': bg_palette,
                'font_sets': font_sets_local,
                'font_id_base': 0 if (frame_idx % 2 == 0) else 2,
                'lut_id': 0,
            })
    else:
        # Original single-pass: per-frame medoids without coherence
        for frame_idx, image_file in enumerate(image_files):
            print(f"Processing frame {frame_idx + 1}/{len(image_files)}: {os.path.basename(image_file)}")
            image = load_and_scale_image(image_file)
            frame_tiles = extract_frame_tiles(image)

            clustered_medoids = derive_medoids_from_tiles(frame_tiles, 512, palette,
                fg_palette=fg_palette, bg_palette=bg_palette)
            if denoise > 0:
                clustered_medoids = denoise_medoid_set(clustered_medoids, denoise)
            font_sets_local = [clustered_medoids[i * 256:(i + 1) * 256] for i in range(2)]

            tile_assignments = assign_tiles_to_medoids(
                frame_tiles, fg_palette, clustered_medoids, high_quality=high_quality,
                bg_palette=bg_palette, four_color_luts=four_color_luts,
                fg_lut_palettes=fg_palettes, bg_lut_palettes=bg_palettes)

            frames_data.append({
                'width': 640,
                'height': 480,
                'use_single_row_blocks': False,
                'memtext': True,
                'tile_assignments': tile_assignments,
                'palette': fg_palette,
                'bg_palette': bg_palette,
                'font_sets': font_sets_local,
                'font_id_base': 0 if (frame_idx % 2 == 0) else 2,
                'lut_id': 0,
            })

    # Store global palette in first frame for output
    frames_data[0]['global_palette'] = fg_palette
    frames_data[0]['global_bg_palette'] = bg_palette
    if four_color_luts and fg_palettes is not None and bg_palettes is not None:
        frames_data[0]['lut_palettes'] = {'fg': fg_palettes, 'bg': bg_palettes}

    return frames_data


def process_animation_frames_memtext_frame(input_dir, n_coherence=0, high_quality=False, split_palette=False, denoise=0):
    """Process animation frames in MEMTEXT frame mode.

    Frame mode characteristics:
    - Per-frame 256/512 color palette (chunk type 0x01)
    - Per-frame 512 medoids split into 2 font sets of 256 (chunk type 0x02)
    - LUT chunk IDs alternate between pairs (0,2) and (1,3) for double buffering
    - Font set IDs alternate between pairs (0,1) and (2,3)
    - Optional coherence medoids for cross-frame temporal stability
    """
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

    frames_data = []

    if n_coherence > 0:
        # Two-pass: scan all frames for canonical patterns, compute coherence, then process
        # Font bit patterns are palette-independent (luminance-based), so we use a
        # neutral palette for the pre-scan pass.
        neutral_palette = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            neutral_palette[i] = [i, i, i]

        print("Pass 1: Extracting canonical patterns from all frames...")
        all_scaled_images = []
        all_frame_tiles = []
        all_frame_canonical_sets = []
        for frame_idx, image_file in enumerate(image_files):
            print(f"  Pre-scan frame {frame_idx + 1}/{len(image_files)}: {os.path.basename(image_file)}")
            image = load_and_scale_image(image_file)
            frame_tiles = extract_frame_tiles(image)
            all_scaled_images.append(image)
            all_frame_tiles.append(frame_tiles)
            canonical_set = extract_canonical_patterns(frame_tiles, neutral_palette)
            all_frame_canonical_sets.append(canonical_set)

        coherence_medoids = compute_coherence_medoids(
            all_frame_canonical_sets, n_coherence)

        print("Pass 2: Per-frame processing with coherence...")
        for frame_idx in range(len(image_files)):
            print(f"  Processing frame {frame_idx + 1}/{len(image_files)}...")
            image = all_scaled_images[frame_idx]
            frame_tiles = all_frame_tiles[frame_idx]
            frame_canonical = all_frame_canonical_sets[frame_idx]

            # Per-frame palette derivation (256 colors, or split 2x256=512)
            palette = create_memtext_256_color_palette_from_image(image)
            if split_palette:
                fg_pal, bg_pal = create_memtext_512_color_palette_from_image(image)
            else:
                fg_pal = bg_pal = palette

            combined_medoids = derive_medoids_with_coherence(
                frame_canonical, 512, coherence_medoids)
            if denoise > 0:
                combined_medoids = denoise_medoid_set(combined_medoids, denoise)
            font_sets_local = [combined_medoids[i * 256:(i + 1) * 256] for i in range(2)]

            tile_assignments = assign_tiles_to_medoids(
                frame_tiles, fg_pal, combined_medoids, high_quality=high_quality,
                bg_palette=bg_pal)

            frames_data.append({
                'width': 640,
                'height': 480,
                'use_single_row_blocks': False,
                'memtext': True,
                'tile_assignments': tile_assignments,
                'palette': fg_pal,
                'bg_palette': bg_pal,
                'font_sets': font_sets_local,
                'font_id_base': 0 if (frame_idx % 2 == 0) else 2,
                'lut_id': frame_idx % 2,
            })
    else:
        # Original single-pass: per-frame everything
        for frame_idx, image_file in enumerate(image_files):
            print(f"Processing frame {frame_idx + 1}/{len(image_files)}: {os.path.basename(image_file)}")
            image = load_and_scale_image(image_file)

            if split_palette:
                fg_pal, bg_pal = create_memtext_512_color_palette_from_image(image)
                if MEDOID_METHOD != 'legacy':
                    palette = merge_palettes_for_medoids([fg_pal], [bg_pal])
                else:
                    palette = fg_pal
            else:
                palette = create_memtext_256_color_palette_from_image(image)
                fg_pal = bg_pal = palette
            frame_tiles = extract_frame_tiles(image)

            clustered_medoids = derive_medoids_from_tiles(
                frame_tiles, 512, palette,
                fg_palette=fg_pal, bg_palette=bg_pal)
            if denoise > 0:
                clustered_medoids = denoise_medoid_set(clustered_medoids, denoise)
            font_sets_local = [clustered_medoids[i * 256:(i + 1) * 256] for i in range(2)]

            tile_assignments = assign_tiles_to_medoids(
                frame_tiles, fg_pal, clustered_medoids, high_quality=high_quality,
                bg_palette=bg_pal)

            frames_data.append({
                'width': 640,
                'height': 480,
                'use_single_row_blocks': False,
                'memtext': True,
                'tile_assignments': tile_assignments,
                'palette': fg_pal,
                'bg_palette': bg_pal,
                'font_sets': font_sets_local,
                'font_id_base': 0 if (frame_idx % 2 == 0) else 2,
                'lut_id': frame_idx % 2,
            })

    return frames_data


def optimize_character_memtext(char_img, palette, palette_oklab=None, use_oklab=False):
    """Optimize a character tile using a palette.
    Returns (font_def, fg_idx, bg_idx) where font_def is 8 bytes.

    Args:
        char_img: 8x8x3 RGB pixel array
        palette: (N,3) color palette (uint8)
        palette_oklab: optional precomputed Oklab palette (N,3) float32
        use_oklab: if True, use Oklab distances for nearest-color matching
    """
    # Flatten to 64 pixels
    pixels = char_img.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)

    # Whether the palette reserves index 0 for transparency (classic 256 palette)
    has_transparency_index = (palette.shape[0] == 256)

    # If requested, precompute Oklab palette
    if use_oklab and palette_oklab is None:
        try:
            palette_oklab = rgb_to_oklab(palette.astype(np.float32))
        except Exception:
            palette_oklab = None

    # Find two most distinct colors in the tile using luminance thresholding
    unique_colors = np.unique(pixels, axis=0)
    if len(unique_colors) < 2:
        # Uniform tile
        avg_color = pixels.mean(axis=0)
        # Find closest palette color
        if use_oklab and (palette_oklab is not None):
            center_ok = rgb_to_oklab(avg_color.reshape(1, 3)).reshape(3)
            dists = np.linalg.norm(palette_oklab - center_ok, axis=1)
        else:
            dists = np.linalg.norm(palette_f - avg_color, axis=1)
        if has_transparency_index:
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

    # Map centers to closest palette colors using either RGB or Oklab
    if use_oklab and (palette_oklab is not None):
        fg_ok = rgb_to_oklab(fg_center.reshape(1, 3)).reshape(3)
        bg_ok = rgb_to_oklab(bg_center.reshape(1, 3)).reshape(3)
        fg_dists = np.linalg.norm(palette_oklab - fg_ok, axis=1)
        bg_dists = np.linalg.norm(palette_oklab - bg_ok, axis=1)
    else:
        fg_dists = np.linalg.norm(palette_f - fg_center, axis=1)
        bg_dists = np.linalg.norm(palette_f - bg_center, axis=1)

    if has_transparency_index:
        fg_dists[0] = float('inf')  # Skip transparent

    fg_idx = int(np.argmin(fg_dists))
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

def save_output_bin_memtext(frames_data, output_bin, frame_duration=6, use_rle=False, encoding_mode='global', four_color_luts=False):
    """Save binary output in MEMTEXT format.

    Key differences from regular format:
    - Text Color LUT: 1024 bytes per LUT (256 entries × 4 bytes BGRA). FG and BG palettes are emitted as separate LUT chunks.
    - Text Font Data: 4 chunks of 2048 bytes each (font sets 0-3)
    - Per frame: 5 character chunks, 5 color chunks (RLE if use_rle=True, fixed otherwise)
    """
    print(f"Saving MEMTEXT binary to {output_bin}...")
    
    if encoding_mode in ('global', 'hybrid'):
        palette = frames_data[0]['global_palette']
        bg_palette_global = frames_data[0].get('global_bg_palette', palette)
        if encoding_mode == 'global':
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

        # Frame mode cannot support four-color-luts (double-buffering required).
        if encoding_mode == 'frame' and four_color_luts:
            print("Warning: --four-color-luts is not supported with frame encoding; disabling for output.")
            four_color_luts = False
        
        if encoding_mode in ('global', 'hybrid'):
            # If four-color-luts mode is enabled, frames_data[0] may carry
            # explicit LUT palettes under key 'lut_palettes' as a dict:
            #   {'fg': [fg0, fg1], 'bg': [bg0, bg1]}
            if four_color_luts and frames_data and 'lut_palettes' in frames_data[0]:
                lut_palettes = frames_data[0]['lut_palettes']
                # Emit one‑LUT chunks (1024 bytes each). Maintain ordering:
                # FG entries -> chunk IDs 0..(n_fg-1), BG entries -> chunk IDs 2..(2+n_bg-1)
                fg_list = list(lut_palettes.get('fg', []))
                bg_list = list(lut_palettes.get('bg', []))
                pal_list = fg_list + bg_list
                for idx, pal in enumerate(pal_list):
                    pal_bgra = np.zeros((256, 4), dtype=np.uint8)
                    pal_bgra[:, 0] = pal[:, 2]
                    pal_bgra[:, 1] = pal[:, 1]
                    pal_bgra[:, 2] = pal[:, 0]
                    pal_bgra[:, 3] = 255
                    pal_bgra[0, 3] = 0
                    lut_data = pal_bgra.tobytes()
                    chunk_id = palette_list_index_to_chunk_id(idx, len(fg_list))
                    f.write(struct.pack('BBH', 0x01, chunk_id & 0xFF, len(lut_data)))
                    f.write(lut_data)
            else:
                # Emit separate 1-LUT chunks (1024 bytes each): FG -> id 0, BG -> id 2
                fg_bgra = np.zeros((256, 4), dtype=np.uint8)
                fg_bgra[:, 0] = palette[:, 2]
                fg_bgra[:, 1] = palette[:, 1]
                fg_bgra[:, 2] = palette[:, 0]
                fg_bgra[:, 3] = 255
                fg_bgra[0, 3] = 0

                bg_bgra = np.zeros((256, 4), dtype=np.uint8)
                bg_bgra[:, 0] = bg_palette_global[:, 2]
                bg_bgra[:, 1] = bg_palette_global[:, 1]
                bg_bgra[:, 2] = bg_palette_global[:, 0]
                bg_bgra[:, 3] = 255
                bg_bgra[0, 3] = 0

                lut_data_fg = fg_bgra.tobytes()
                lut_data_bg = bg_bgra.tobytes()

                # Emit FG (index 0) and BG (index 1) using canonical mapping
                f.write(struct.pack('BBH', 0x01, palette_list_index_to_chunk_id(0, 1) & 0xFF, len(lut_data_fg)))
                f.write(lut_data_fg)
                f.write(struct.pack('BBH', 0x01, palette_list_index_to_chunk_id(1, 1) & 0xFF, len(lut_data_bg)))
                f.write(lut_data_bg)

            if encoding_mode == 'global':
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

            if encoding_mode == 'frame':
                frame_fg_pal = frame_data['palette']
                frame_bg_pal = frame_data.get('bg_palette', frame_fg_pal)
                frame_font_sets = frame_data['font_sets']
                lut_id = frame_data.get('lut_id', frame_idx % 2)
                font_id_base = frame_data.get('font_id_base', 0 if (frame_idx % 2 == 0) else 2)
                # Per-frame LUTs: in four-color mode the frame may provide
                # 'lut_palettes' with {'fg': [fg0, fg1], 'bg': [bg0, bg1]}
                # Frame-mode: emit two 1-LUT chunks per frame (double-buffered)
                # FG -> chunk (lut_id & 1), BG -> chunk (2 + (lut_id & 1))
                fg_bgra = np.zeros((256, 4), dtype=np.uint8)
                fg_bgra[:, 0] = frame_fg_pal[:, 2]
                fg_bgra[:, 1] = frame_fg_pal[:, 1]
                fg_bgra[:, 2] = frame_fg_pal[:, 0]
                fg_bgra[:, 3] = 255
                fg_bgra[0, 3] = 0

                bg_bgra = np.zeros((256, 4), dtype=np.uint8)
                bg_bgra[:, 0] = frame_bg_pal[:, 2]
                bg_bgra[:, 1] = frame_bg_pal[:, 1]
                bg_bgra[:, 2] = frame_bg_pal[:, 0]
                bg_bgra[:, 3] = 255
                bg_bgra[0, 3] = 0

                fg_chunk_id, bg_chunk_id = lut_id_to_fg_bg_chunk_ids(lut_id)

                lut_data_fg = fg_bgra.tobytes()
                lut_data_bg = bg_bgra.tobytes()

                f.write(struct.pack('BBH', 0x01, fg_chunk_id & 0xFF, len(lut_data_fg)))
                f.write(lut_data_fg)
                f.write(struct.pack('BBH', 0x01, bg_chunk_id & 0xFF, len(lut_data_bg)))
                f.write(lut_data_bg)

                for local_fs in range(2):
                    font_data = frame_font_sets[local_fs].astype(np.uint8).tobytes()
                    if len(font_data) < 2048:
                        font_data += b'\x00' * (2048 - len(font_data))
                    f.write(struct.pack('BBH', 0x02, (font_id_base + local_fs) & 0x03, len(font_data)))
                    f.write(font_data)
            elif encoding_mode == 'hybrid':
                # Hybrid: no per-frame palette, only per-frame font sets
                frame_font_sets = frame_data['font_sets']
                font_id_base = frame_data.get('font_id_base', 0 if (frame_idx % 2 == 0) else 2)
                lut_id = 0  # Always LUT 0 (FPGA LUT 1 bug workaround)

                for local_fs in range(2):
                    font_data = frame_font_sets[local_fs].astype(np.uint8).tobytes()
                    if len(font_data) < 2048:
                        font_data += b'\x00' * (2048 - len(font_data))
                    f.write(struct.pack('BBH', 0x02, (font_id_base + local_fs) & 0x03, len(font_data)))
                    f.write(font_data)
            else:
                font_id_base = 0
                lut_id = 0
            
            tile_assignments = frame_data['tile_assignments']  # (4800, 5)
            
            # Build character data: 2 bytes per tile
            # Encoding: byte0 = char_index, byte1 = (font_set:2, invert:1, unused:5)
            # Actually, let's use: word = char_index | (font_set << 8) | (invert << 10)
            # For MEMTEXT per FileFormat.md, RLE encodes words (2 bytes) not single bytes
            
            char_data = []
            for i in range(len(tile_assignments)):
                fs = tile_assignments[i, 0]
                if encoding_mode in ('frame', 'hybrid'):
                    fs = (font_id_base + fs) & 0x03
                char_idx = tile_assignments[i, 1]
                invert = int(tile_assignments[i, 4])

                # Per spec bits mapping — extended for four-color-luts
                # bits [8:9] = font bank
                # bit 10 = FG LUT select
                # bit 11 = BG LUT select
                # bit 12 = invert
                if four_color_luts and getattr(tile_assignments, 'shape', None) and tile_assignments.shape[1] >= 7:
                    fg_lut_bit = int(tile_assignments[i, 5]) & 0x01
                    bg_lut_bit = int(tile_assignments[i, 6]) & 0x01
                else:
                    lut_sel = lut_id & 0x01
                    fg_lut_bit = lut_sel
                    bg_lut_bit = lut_sel

                high_byte = (
                    (fs & 0x03)
                    | ((fg_lut_bit & 0x01) << 2)
                    | ((bg_lut_bit & 0x01) << 3)
                    | ((invert & 0x01) << 4)
                )
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

