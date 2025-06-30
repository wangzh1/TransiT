import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

time_bins = np.arange(0, 1000, 1) 
simulated_histogram = np.zeros((1000, 64, 64))

peak_position = 500       
peak_width = 50          
peak_amplitude = 1000      
gaussian_peak = peak_amplitude * np.exp(-0.5 * ((time_bins - peak_position) / peak_width) ** 2)
simulated_histogram = np.tile(gaussian_peak[:, np.newaxis, np.newaxis], (1, 64, 64))


noisy_histogram = np.random.poisson(simulated_histogram)

dark_counts_rate = 0.1        
background_light_rate = 0.05  

total_background_rate = dark_counts_rate + background_light_rate
background_counts = np.random.poisson(total_background_rate, size=simulated_histogram.shape)
noisy_histogram += background_counts
jitter_std = 5 
irf_bins = np.arange(-50, 51, 1) 
irf = np.exp(-0.5 * (irf_bins / jitter_std) ** 2)
irf = irf / irf.sum() 
noisy_histogram_convolved = fftconvolve(noisy_histogram, irf[:, np.newaxis, np.newaxis], mode='same', axes=0)


pixel_x = 32 
pixel_y = 32

plt.figure(figsize=(10, 6))
plt.plot(time_bins, simulated_histogram[:, pixel_x, pixel_y], label='simu')
plt.plot(time_bins, noisy_histogram_convolved[:, pixel_x, pixel_y], label='noise')
plt.xlabel('time')
plt.ylabel('count')
plt.legend()
plt.title(f'NLOS histogram - ({pixel_x}, {pixel_y})')
plt.savefig(f'nlos_histogram_pixel_{pixel_x}_{pixel_y}.png', dpi=300)