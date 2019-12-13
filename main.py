import cv2 as cv2
import sys
from color_transfer.commonfunctions import *
from color_transfer.color_transfer import *
from domain_transform.domain_transform import *
from pca.pca import *
from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.util import view_as_windows, pad, random_noise
from skimage.segmentation import *
from scipy.ndimage import binary_fill_holes
from timeit import default_timer as timer
from edge_segmentation.edge_segmentation import *
from face_segmentation.face_segmentation import *
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt

LMAX = 3
IM_SIZE = 400
PATCH_SIZES = np.array([33, 21, 13, 9, 5])
SAMPLING_GAPS = np.array([28, 18, 8, 5, 3])
IALG = 10
IRLS_it = 3
IRLS_r = 0.8
PADDING_MODE = 'edge'


def build_gaussian_pyramid(img, L):
    img_arr = []
    img_arr.append(img)  # D_L (img) = img
    for i in range(L - 1):
        img_arr.append(cv2.pyrDown(img_arr[-1].astype(np.float32)).astype(np.float32))
    return img_arr


def get_segmentation_mask(mode, img=None, c=1.0):
    if mode == 'none' or mode is None or img is None:
        return np.ones((IM_SIZE, IM_SIZE), dtype=np.float32) * c
    elif mode == 'edge':
        return edge_segmentation(img) * c
    elif mode == 'face':
        return segment_faces(img) * c


def solve_irls(X, X_patches_raw, p_index, style_patches, neighbors, projection_matrix):
    p_size = PATCH_SIZES[p_index]
    sampling_gap = SAMPLING_GAPS[p_index]
    current_size = X.shape[0]
    # Extracting Patches
    X_patches = X_patches_raw.reshape(-1, p_size * p_size * 3)
    npatches = X_patches.shape[0]
    if p_size <= 21:
        X_patches = project(X_patches, projection_matrix)  # Projecting X to same dimention as style patches
    # Computing Nearest Neighbors
    distances, indices = neighbors.kneighbors(X_patches)
    distances += 0.0001
    # Computing Weights
    weights = np.power(distances, IRLS_r - 2)
    # Patch Accumulation
    R = np.zeros((current_size, current_size, 3), dtype=np.float32)
    Rp = extract_patches(R, patch_shape=(p_size, p_size, 3), extraction_step=sampling_gap)
    X[:] = 0
    t = 0
    for t1 in range(X_patches_raw.shape[0]):
        for t2 in range(X_patches_raw.shape[1]):
            nearest_neighbor = style_patches[indices[t, 0]]
            X_patches_raw[t1, t2, 0, :, :, :] += nearest_neighbor * weights[t]
            Rp[t1, t2, 0, :, :, :] += 1 * weights[t]
            t = t + 1
    R += 0.0001  # to avoid dividing by zero.
    X /= R




def style_transfer(content, style, segmentation_mask, sigma_r=0.17, sigma_s=15):
    #down sample for LMAX times
    content_arr = build_gaussian_pyramid(content, LMAX) 
    style_arr = build_gaussian_pyramid(style, LMAX)
    segm_arr = build_gaussian_pyramid(segmentation_mask, LMAX) #segmentation mask

    # Initialize X with the content + strong noise.
    X = random_noise(content_arr[LMAX - 1], mode='gaussian', var=50)
    
    # Set up Content Fusion constants.
    fus_const1 = []
    fus_const2 = []
    for i in range(LMAX):
        sx, sy = segm_arr[i].shape
        curr_segm = segm_arr[i].reshape(sx, sy, 1) #???
        fus_const1.append(curr_segm * content_arr[i]) #(W*C)
        fus_const2.append(1.0 / (curr_segm + 1)) #(W+I)^-1
    print('Starting Style Transfer..')
    for L in range(LMAX - 1, -1, -1):  # over scale L
        print('Scale ', L)
        current_size = style_arr[L].shape[0]
        style_L_sx, style_L_sy, _ = style_arr[L].shape
        X = random_noise(X, mode='gaussian', var=20 / 250.0)

        #patch matching 
        for n in range(PATCH_SIZES.size):  # over patch size n
            p_size = PATCH_SIZES[n]
            print('Patch Size', p_size)
            npatchx = int((style_L_sx - p_size) / SAMPLING_GAPS[n] + 1)	
            # The images are padded to avoid side artifacts.
            padding = p_size - (style_L_sx - npatchx * SAMPLING_GAPS[n])
            padding_arr = ((0, padding), (0, padding), (0, 0))
            current_style = pad(style_arr[L], padding_arr, mode=PADDING_MODE)
            X = pad(X, padding_arr, mode=PADDING_MODE)
            const1 = pad(fus_const1[L], padding_arr, mode=PADDING_MODE)
            const2 = pad(fus_const2[L], padding_arr, mode=PADDING_MODE)
            style_patches = extract_patches(current_style, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
            npatchx, npatchy, _, _, _, _ = style_patches.shape
            npatches = npatchx * npatchy
            # Preparing for NN
            style_patches = style_patches.reshape(-1, p_size * p_size * 3)
            njobs = 1
            if (L == 0) or (L == 1 and p_size <= 13):
                njobs = -1
            projection_matrix = 0
            # for small patch sizes perform PCA
            if p_size <= 21:
                new_style_patches, projection_matrix = pca(style_patches) #reduce dimensions of patches
                neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(new_style_patches) #sklearn package
            else:
                neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(style_patches)
            style_patches = style_patches.reshape((-1, p_size, p_size, 3))


            for k in range(IALG): #loop over patch sizes
                # Steps 1 & 2: Patch-Extraction and and Robust Patch Aggregation
                X_patches_raw = extract_patches(X, patch_shape=(p_size, p_size, 3), extraction_step=SAMPLING_GAPS[n])
                for i in range(IRLS_it):
                    solve_irls(X, X_patches_raw, n, style_patches, neighbors, projection_matrix)
                # Step 3: Content Fusion
                X = const2 * (X + const1) #formula 15
                # Step 4: Color Transfer
                X = color_transfer(X, style)
                # Step 5: Denoising
                X[:style_L_sx, :style_L_sx, :] = denoise(X[:style_L_sx, :style_L_sx, :], sigma_r=sigma_r, sigma_s=sigma_s)
            X = X[:style_L_sx, :style_L_sx, :]  # Discard padding.
        # Upscale X
        if (L > 0):
            sizex, sizey, _ = content_arr[L - 1].shape
            X = cv2.resize(X, (sizex, sizey))
    return X

def implement_transfer(content_img, style_img, seg_mode = 'edge'):
    """
    content_img, style_img -- strings representing image file name
    """
    content = io.imread('images/'+content_img) / 255.0
    style = io.imread('images/paper_images/' + style_img) / 255.0
    segm_mask = edge_segmentation(content, 5, 0.6)
    content = (cv2.resize(content, (IM_SIZE, IM_SIZE))).astype(np.float32)
    style = (cv2.resize(style, (IM_SIZE, IM_SIZE))).astype(np.float32)
    segm_mask = (cv2.resize(segm_mask, (IM_SIZE, IM_SIZE))).astype(np.float32)
    #show_images([content, segm_mask, style])
    #original_content = content.copy()
    content = color_transfer(content, style)
    start = timer()
    X = style_transfer(content, style, segm_mask)
    end = timer()
    print("Style Transfer took ", end - start, " seconds!")
    # Finished. Just save the images
    plt.imshow(X)
    plt.savefig('./transfered_img/'+content_img[:-4]+'-'+style_img[:-4]+'_transfered.jpg')
    return 


def main():
    
    if len(sys.argv) == 3:
        implement_transfer(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        implement_transfer(sys.argv[1], sys.argv[2], seg_mode = sys.argv[3])
    else:
        print('Wrong input. Check the command')
    
    


main()

