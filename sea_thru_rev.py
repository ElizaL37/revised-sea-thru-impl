# 
# This file was adapted from Sea-Thru-Imp ->  Copyright (c) 2022 Zeyuan HE (Teragion).
#
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import argparse
import ctypes
import os
import numpy as np  
import time
import scipy
import scipy.optimize

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# loads a precompiled library (sillu), compiled with 'illuminant_v2.cpp'
# to complie new file for use in this script: g++ -O2 -fopenmp -shared -fPIC -o sillu.so illuminant_v2.cpp
lib = np.ctypeslib.load_library('sillu','.')    


NUM_BINS = 10 # number of bins of depths to find backscatter (def. Sea-Thru Sec. 4.3)


def read_image(image_path, max_side = 3840):
    # Opens png files
    image_file = Image.open(image_path).convert("RGB")
    image_file.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(image_file).astype(np.float64) / 255.0


def read_depthmap(depthmap_path, size):
    depth_file = Image.open(depthmap_path)
    depths = depth_file.resize(size, Image.Resampling.LANCZOS)
    return np.float64(depths)


"""
Preprocessing - makes sure there are no '0's or invalid depths
"""
def normalize_depth_map(depths, z_min, z_inf):
    z_max = max(np.max(depths), z_inf)
    depths[depths == 0] = z_max
    depths[depths < z_min] = z_min
    return depths


"""
Convert sRGB gamma-encoded image to linear RGB.
Reference: IEC 61966-2-1:1999 (sRGB standard)
Found Method:  https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
"""
def srgb_to_linear(image):
    a = 0.04045
    linear = np.empty_like(image)
    low_mask = image <= a

    # Linear segment for low intensities
    linear[low_mask] = image[low_mask] / 12.92

    # Power-law segment for higher intensities
    linear[~low_mask] = ((image[~low_mask] + 0.055) / 1.055) ** 2.4
    return linear


"""
Converts a linear RGB image back to sRGB
Used for output display images
"""
def linear_to_srgb(linear_image):

    a = 0.0031308
    srgb = np.empty_like(linear_image)
    low_mask = linear_image <= a
    srgb[low_mask] = linear_image[low_mask] * 12.92
    srgb[~low_mask] = 1.055 * (linear_image[~low_mask] ** (1/2.4)) - 0.055
    return np.clip(srgb, 0, 1) # Clip to ensure valid range

# Backscatter Calculations ######################################################

"""
Finds the darkest pixels in the image for various depth ranges.    
This implements the logic from Section 4.3 of Sea-Thru Paper by Akkaynak & Treibitz 
"""
def find_reference_points_darkest(image, depths, percentile = 1):

    valid_mask = np.isfinite(depths) & (depths > 0)
    valid_depths = depths[valid_mask]
    valid_colour_pixels = image[valid_mask]

    z_max = np.max(valid_depths)
    z_min = np.min(valid_depths)

    z_bins = np.linspace(z_min, z_max, NUM_BINS + 1)
    bin_indices = np.digitize(valid_depths, z_bins) # from: https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    bin_indicies = np.minimum(bin_indices, NUM_BINS)

    ref_points = [] # bottom 1% darkest pixels from each bin
    for i in range(1, NUM_BINS + 1):
        
        bin_mask = (bin_indices == i)
        curr_colour_pixels = valid_colour_pixels[bin_mask]
        curr_depths = valid_depths[bin_mask]

        if curr_colour_pixels.shape[0] == 0:
            continue

        # To find darkest pixel sum rgb channels : "we search for the darkest RGB triplets rather than finding the darkest pixels independently in each color channel and we do not form a dark channel image"
        rgb_sum = np.sum(curr_colour_pixels, axis=1)
        bott_threshold = np.percentile(rgb_sum, percentile)

        if rgb_sum.shape[0] == 0:
            continue

        dark_mask = (rgb_sum < bott_threshold)
        dark_depths = curr_depths[dark_mask]

        if dark_depths.shape[0] == 0:
            continue

        ref_points.append(np.hstack((dark_depths[:, np.newaxis], curr_colour_pixels[dark_mask])))

    return np.concatenate(ref_points, axis = 0)


"""
    Cost Function Eq. 10 - Backscatter Estimation, including the residual term (if selected in cmd ln args).
"""
def residual_backscatter(params, z, data):

    Bc_inf, betaB, Jc_dash, betaD_dash = params    
    prediction = (Bc_inf * (1 - np.exp(-betaB * z)) + Jc_dash * np.exp(-betaD_dash * z))
    
    # Return the error
    return prediction - data


"""
    Fit Backscatter at ifinity & the backscatter coefficient
"""
def estimate_channel_backscatter(points, depths, channel, attempts = 20):

    lo = np.array([0, 0, 0, 0])
    hi = np.array([1, 5, 1, 5])

    best_loss = np.inf
    best_coeffs = [0, 0, 0, 0]

    x_data = points[:, 0]
    y_data = points[:, channel + 1]

    for _ in range(attempts):                  
        initial_guess = np.random.random(4) * (hi - lo) + lo

        # How to implement from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        result = scipy.optimize.least_squares(residual_backscatter, 
                                              initial_guess, 
                                              bounds=(lo, hi), 
                                              loss='soft_l1', 
                                              args=(x_data, y_data), 
                                              max_nfev=1000)

        if result.cost < best_loss:
            best_loss = result.cost
            best_coeffs = result.x    

    print(f"Found coeffs for channel {channel} with loss {best_loss:.4f}")
    print(f"Bc_inf = {best_coeffs[0]:.5f}, betaB={best_coeffs[1]:.5f}, J_dash={best_coeffs[2]:.5f}, betaD_dash={best_coeffs[3]:.5f}")

    return best_coeffs


"""
Calculates the final backscatter map for the entire image.    
"""
def calculate_final_backscatter(depths, B_inf, beta_B, J_prime, beta_D_prime, use_residual=True):

    output_shape = (depths.shape[0], depths.shape[1], 3)
    B_map = np.zeros(output_shape, dtype=np.float64)

    for c in range(3):
        B_map[..., c] = B_inf[c] * (1 - np.exp(-beta_B[c] * depths))
        if use_residual:
            #print("\nBackscatter estimate includes residual term J'·exp(-βD'·z)")
            B_map[..., c] += J_prime[c] * np.exp(-beta_D_prime[c] * depths)

    invalid_mask = ~np.isfinite(depths) | (depths <= 0)
    B_map[invalid_mask] = 0

    return B_map


"""
    Main Function for Backscatter estimation

    Finds dark reference points,
    then simultaneously executes estimations for channel backscatter,
    then forms final backscatter image.

"""
def estimate_backscatter(image, depths, use_residual=True):

    print("\nFind darkest reference points")
    ref_points = find_reference_points_darkest(image, depths)

    backscatter_coeffs = [None] * 3

    print("\nEstimate backscatter parameters")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures_for_channels = {
            executor.submit(estimate_channel_backscatter, ref_points, depths, colour_channel): colour_channel
            for colour_channel in range(3)
        }
        for future in as_completed(futures_for_channels):
            curr_channel = futures_for_channels[future]
            backscatter_coeffs[curr_channel] = future.result()

    B_inf        = np.array([coeffs[0] for coeffs in backscatter_coeffs])
    beta_B       = np.array([coeffs[1] for coeffs in backscatter_coeffs])
    J_prime      = np.array([coeffs[2] for coeffs in backscatter_coeffs])
    beta_D_prime = np.array([coeffs[3] for coeffs in backscatter_coeffs])

    print("\nCalculating final backscatter map")
    Ba = calculate_final_backscatter(depths, B_inf, beta_B, J_prime, beta_D_prime, use_residual=use_residual)

    return Ba, backscatter_coeffs

# Attenuation Calculations ######################################################

"""
    Curve for predicting wideband attenuation per pixel, given depth and fitted parameters a,b,c,d   
"""
def predict_wideband_attenuation(depths, a, b, c, d):
    
    return a * np.exp(b * depths) + c * np.exp(d * depths)


"""
    
    Returns z - z_hat per pixel
 """
def residuals_diff(params, depths_fit, neg_log_Ec):    
    a, b, c, d = params
    beta_D = a * np.exp(b * depths_fit) + c * np.exp(d * depths_fit)
    beta_D = np.maximum(beta_D, 1e-6)
    z_hat = neg_log_Ec / beta_D
    return depths_fit - z_hat


"""
    AttenuationFitting function
    Minimises ||z - z_hat|| (Eq. 17), restricted to pixels inside valid_mask.
    Invalid/noisy depth pixels are excluded from the fit.
"""
def fit_attenuation_channel(channel, Ea, depths, valid_mask, coarse_attempts, max_fit_points=100000):
    
    Ec = Ea[:, :, channel]
    original_shape = Ec.shape

    fit_mask = valid_mask & (Ec > 1e-5) & (Ec < 0.999)
    locs_fit = np.where(fit_mask)
    Ec_fit = Ec[locs_fit]
    depths_fit = depths[locs_fit]


    if Ec_fit.size > max_fit_points:
        indices = np.random.choice(Ec_fit.size, max_fit_points, replace=False)
        Ec_fit = Ec_fit[indices]
        depths_fit = depths_fit[indices]

    neg_log_Ec = -np.log(Ec_fit)

    # Coarse estimate of beta_D per pixel via Eq. 12: beta_D_hat = -log(Ec) / z.
    # "bounds can be narrowed using the coarse estimate obtained from Eq. 12."
    beta_D_hat = neg_log_Ec / depths_fit
    coarse_valid = (beta_D_hat > 0.0) & (beta_D_hat < 5.0)
    if coarse_valid.sum() > 10:
        beta_D_hat_valid = beta_D_hat[coarse_valid]
        amp_hi = float(np.percentile(beta_D_hat_valid, 90))
        beta_D_median = float(np.median(beta_D_hat_valid))
    else:
        amp_hi = 5.0
        beta_D_median = 1.0
    print(f"  Coarse beta_D_hat for ch{channel}: median={beta_D_median:.4f}, 90th pctl={amp_hi:.4f}")

    
    b_lo = np.array([0.0,    -10.0,   0.0,    -10.0])
    b_hi = np.array([amp_hi, 0, amp_hi, 0])

    # First guess initialisation from the coarse estimate
    initilised_guess = np.array([beta_D_median / 2.0, -1.0, beta_D_median / 2.0, -0.1])

    best_loss = np.inf
    best_coeffs = np.clip(initilised_guess, b_lo, b_hi)

    for i in range(coarse_attempts):
        if i == 0:
            initial_guess = np.clip(initilised_guess, b_lo, b_hi)
        else:
            initial_guess = np.random.uniform(b_lo, b_hi)
        try:
            result = scipy.optimize.least_squares(residuals_diff, 
                                                  initial_guess,
                                                  args=(depths_fit, neg_log_Ec),
                                                  bounds=(b_lo, b_hi),
                                                  loss='linear',
                                                  max_nfev=1000)
            popt = result.x

        except (ValueError, RuntimeError) as e:
            print(f"Fit failed for channel {channel} attempt {i+1}: {e}")
            continue

        cur_loss = np.mean(np.square(residuals_diff(popt, depths_fit, neg_log_Ec)))
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_coeffs = popt

    print(f"Section 4.4.3 fit for channel {channel} — depth MSE: {best_loss:.4f}")
    print(f"a = {best_coeffs[0]:.4f}, b = {best_coeffs[1]:.4f}, c = {best_coeffs[2]:.4f}, d = {best_coeffs[3]:.4f}")

    # Predict beta_D for all valid pixels
    locs_all = np.where(valid_mask & (Ec > 1e-5))
    att_channel = np.zeros(original_shape, dtype=np.float64)
    att_channel[locs_all] = predict_wideband_attenuation(depths[locs_all], *best_coeffs)

    return att_channel


"""
    Main Function for Attenuation estimation

    Calculate illuminant map, in cpp for speed,
    then simultaneously fit attentuation parameters for each chennel

"""
def estimate_wideband_attenuation(D, depths, valid_mask, coarse_attempts=10):
    
    Ea = compute_illuminant_map_plugin(D, depths, iterations = 100, p=0.3, f=2, eps = (np.max(depths) - np.min(depths)) * 0.1)
    Ea = np.clip(Ea, 1e-10, None)

    att_channels = [None] * 3

    # All 3 channel fits at the same time
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_channel = {
            executor.submit(fit_attenuation_channel, channel, Ea, depths, valid_mask, coarse_attempts): channel
            for channel in range(3)
        }

        for future in as_completed(future_to_channel):
            channel = future_to_channel[future]
            try:
                att_channels[channel] = future.result()
            except Exception as exc:
                print(f"Attenuation fit for channel {channel} generated an exception: {exc}")
                att_channels[channel] = np.zeros_like(depths)

    att = np.stack(att_channels, axis=2)

    return att, Ea


"""
    Helper function to run the Cpp-plugin for one channel
"""
def compute_illuminant_channel(channel_data, depths, iterations, p, f, eps):

    simple_illuminant_cpp_func = lib.compute_illuminant_map

    simple_illuminant_cpp_func.restype = None
    simple_illuminant_cpp_func.argtypes = [np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous'),
                     np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous'),
                     np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous, writeable'),
                     ctypes.c_double,
                     ctypes.c_double,
                     ctypes.c_double,
                     ctypes.c_int, 
                     ctypes.c_int,
                     ctypes.c_int]

    Dc = np.ascontiguousarray(channel_data)
    Dc = np.require(Dc, float, ['ALIGNED', 'CONTIGUOUS'])
    ac = np.zeros_like(Dc)
    ac = np.require(ac, float, ['ALIGNED', 'CONTIGUOUS'])
    x, y = depths.shape
    z = np.require(depths, float, ['ALIGNED', 'CONTIGUOUS'])
    simple_illuminant_cpp_func(Dc, z, ac, p, f, eps, x, y, iterations)
    
    return ac

"""
    Calls C interface for computing illuminant map for all 3 channels in parallel, 
    then stacks results into one 3-channel illuminant map.
""" 
def compute_illuminant_map_plugin(D, depths, iterations, p, f, eps):
  
    results = [None] * 3
    with ThreadPoolExecutor(max_workers=3) as executor:
        
        future_to_channel = {
            executor.submit(compute_illuminant_channel, D[:, :, channel], depths, iterations, p, f, eps): channel
            for channel in range(3)
        }
        
        for future in as_completed(future_to_channel):
            channel_index = future_to_channel[future]            
            results[channel_index] = future.result()
                    
    
    return np.stack(results, axis=2)


"""
    Recovers the scene J_c = D_c * exp(B_D(z) * z)
"""
def recover(Da, att, depths):
    
    if depths.ndim == 2:
        depths_3d = depths[..., np.newaxis]
    
    att_z = att * depths_3d
    att_z = np.clip(att_z, 0, 5.0) 
    recovery_map = np.exp(att_z)
    Ja = Da * recovery_map

    valid_mask = np.isfinite(depths) & (depths > 0)
    Ja[~valid_mask] = 0.0
    return Ja

# Post-processing: White Balancing and Normalisation ######################################################
"""
    Applies white balancing
    after Wb Ja normalised into a displayable [0, 1] range.
"""
def post_process(Ja, valid_mask):
    
    print("Post-processing: White Balancing...")

    Ja_valid = Ja[valid_mask] 

    mu = Ja_valid.mean(axis=0)
    overall_mu = mu.mean()
    gw_scale = overall_mu / np.maximum(mu, 1e-6)
    Ja_balanced = Ja * gw_scale
    Ja_balanced_valid = Ja_balanced[valid_mask]
    Wc = float(np.percentile(Ja_balanced_valid, 99))
    Wc = max(Wc, 1e-6)  

    Js = Ja_balanced / Wc
    Js = np.clip(Js, 0, 1)

    Js_valid_after = Js[valid_mask]

    mask_3d = valid_mask[:, :, np.newaxis]
    Js = np.where(mask_3d, Js, 0.0)
    #Js = np.power(Js, 1.15)
    return Js


if __name__ == '__main__':
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--original',    required=True,  help="Path to original image")
    parser.add_argument('--depth',       required=True,  help="Path to depth map")
    parser.add_argument('--size',        required=False, type=int, help="Maximum side length to resize image to")
    parser.add_argument('--prefix',      required=False, help="Prefix for output files")
    parser.add_argument('--output',      required=False, help="Directory to save all output images")
    parser.add_argument('--no-residual', action='store_true',
                        help="Omit residual term J'·exp(-βD'·z) from backscatter model (Eq. 10). "
                             "Use when data does not satisfy the low-scatter conditions described "
                             "in Akkaynak & Treibitz (2019) Section 4.3.")

    args = parser.parse_args()

    original_image = read_image(args.original, args.size) if args.size is not None else read_image(args.original)
    prefix = args.prefix if args.prefix is not None else ""

    depths = read_depthmap(args.depth, (original_image.shape[1], original_image.shape[0]))
    depth_time = time.perf_counter()
    elapsed_time = depth_time - start_time
    
    print("Loaded image and depth map of size {x} x {y}".format(x = original_image.shape[0], y = original_image.shape[1]))

    print("Estimating backscatter...", flush=True)
    original_image_linearised = srgb_to_linear(original_image)
    Ba, coeffs = estimate_backscatter(original_image_linearised, depths, use_residual=not args.no_residual)
    backscatter_time = time.perf_counter()
    elapsed_time = backscatter_time - depth_time
    print(f"\nFinished estimating backscatter: {elapsed_time:.2f} seconds", flush=True)

    Da = original_image_linearised - Ba
    Da = np.clip(Da, 0, 1)

    # For Debugging backscatter

    # Da_sRGB = linear_to_srgb(Da)
    # D = np.uint8(Da_sRGB * 255.0)
    # backscatter_removed = Image.fromarray(D)    
    # bs_path = os.path.join(args.output, prefix + "direct_signal.png") if args.output else "out_test/" + prefix + "direct_signal.png"
    # backscatter_removed.save(bs_path)
    

    print("Estimating wideband attenuation...", flush=True)    
    original_valid_mask = np.isfinite(depths) & (depths > 0)
    depths = normalize_depth_map(depths, 0.1, 6.0)

    # Zero invalid-depth pixels in Da so they cannot influence illuminant or attenuation fits
    Da_valid = Da.copy()
    Da_valid[~original_valid_mask] = 0.0

    att, Ea = estimate_wideband_attenuation(Da_valid, depths, original_valid_mask)
    print(f"  Invalid depth pixels: {(~original_valid_mask).sum()} "
          f"({(~original_valid_mask).mean()*100:.1f}% of image)")
    attenuation_time = time.perf_counter()
    elapsed_time = attenuation_time - backscatter_time
    print(f"\nFinished estimating attenuation: {elapsed_time:.2f} seconds", flush=True)

    # Debug illuminant map uncomment:

    # Ea = np.clip(Ea, 0, 1)
    # Ea_sRGB = linear_to_srgb(Ea)
    # E = np.uint8(Ea_sRGB * 255.0)
    # illuminant_map = Image.fromarray(E)
    # il_path = os.path.join(args.output, prefix + "illuminant_map.png") if args.output else "out_test/" + prefix + "illuminant_map.png"
    # illuminant_map.save(il_path)
    

    Ja = recover(Da_valid, att, depths)

    recover_time = time.perf_counter()
    elapsed_time = recover_time - attenuation_time
    print(f"\nFinished recovery on image: {elapsed_time:.2f} seconds")
   
    Js = post_process(Ja, original_valid_mask)
    Js_sRGB = linear_to_srgb(Js)
    J = np.uint8(Js_sRGB * 255.0)
    
    # Apply transparency for improved 3D reconstruction results
    J_rgba = np.zeros((J.shape[0], J.shape[1], 4), dtype=np.uint8)
    J_rgba[:, :, :3] = J
    J_rgba[:, :, 3] = np.where(original_valid_mask, 255, 0)  # Alpha channel (0=transparent, 255=opaque)
    
    result = Image.fromarray(J_rgba, 'RGBA')
    final_path = os.path.join(args.output, prefix + "final_output.png") if args.output else "out_test/" + prefix + "out.png"
    result.save(final_path)
    print("Finished.")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

