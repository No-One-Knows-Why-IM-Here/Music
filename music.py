import sounddevice as sd
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.cm as cm  # For colormaps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import sys

class SpectrumAnalyzer3D:
    def __init__(self):
        # Create the application
        self.app = QtWidgets.QApplication(sys.argv)

        # Set up the 3D plot window
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('🎵 :D 🎵')
        self.view.setGeometry(100, 100, 800, 600)
        self.view.setBackgroundColor('#000000')  # Black background
        self.view.opts['distance'] = 100  # Adjust the camera distance for better view
        self.view.show()

        # Initialize parameters
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024  # Reduced chunk size for better performance
        self.max_freq = 20000  # Limit frequency to 20kHz

        # Compute frequencies
        freq = np.fft.rfftfreq(self.CHUNK_SIZE, d=1./self.SAMPLE_RATE)
        # Limit frequencies to 20kHz
        self.freq_indices = np.where(freq <= self.max_freq)[0]
        self.freq = freq[self.freq_indices]
        self.num_freq_bins = len(self.freq)

        # Reduce the number of frequency bins to improve performance
        self.desired_bins = 256 # Adjust this number for performance vs. resolution
        self.bin_indices = np.linspace(0, self.num_freq_bins - 1, self.desired_bins).astype(int)
        self.freq = self.freq[self.bin_indices]
        self.num_freq_bins = len(self.freq)

        # Define x and y as frequency axes
        self.x = self.freq  # Frequencies for x-axis
        self.y = self.freq  # Frequencies for y-axis

        # Initialize Z data (outer product of FFT amplitudes)
        self.Z = np.zeros((self.num_freq_bins, self.num_freq_bins))  # Shape: (desired_bins, desired_bins)

        # Create colormap for amplitude
        self.cmap = plt.colormaps['turbo']  # or use plt.get_cmap('turbo')  # Use 'turbo' colormap for more colors

        # Verify the shapes of x, y, and Z
        print(f"x shape: {self.x.shape}, y shape: {self.y.shape}, Z shape: {self.Z.shape}")

        # Create the surface plot
        try:
            self.surface = gl.GLSurfacePlotItem(
                x=self.x,
                y=self.y,
                z=self.Z,
                shader='shaded',  # Use 'shaded' shader to apply custom colors
                smooth=False,
                glOptions='opaque'  # Opaque rendering for better performance
            )
            self.view.addItem(self.surface)
        except Exception as e:
            print(f"Error initializing GLSurfacePlotItem: {e}")
            sys.exit(1)

        # Set up the audio stream
        self.device = self.get_input_device()

        if self.device is None:
            print("Could not find an input device. Please check your audio devices.")
            sys.exit(1)
        else:
            print(f"Using device: {self.device['name']}")

        self.stream = sd.InputStream(
            device=self.device['index'],
            channels=1,
            samplerate=self.SAMPLE_RATE,
            blocksize=self.CHUNK_SIZE,
            callback=self.audio_callback
        )
        self.stream.start()

        # Buffer for audio data
        self.data = np.zeros(self.CHUNK_SIZE)

        # Set up the timer for updating the plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)  # Update every 1 ms

    def get_input_device(self):
        """
        Displays a list of input devices and asks the user to select one via a dialog.
        """
        try:
            devices = sd.query_devices()
            input_devices = [device for device in devices if device['max_input_channels'] > 0]
            if not input_devices:
                print("No input devices found.")
                return None

            # Build a list of device names to display
            device_list = [f"{device['index']}: {device['name']}" for device in input_devices]

            # Create a dialog for device selection
            item, ok = QtWidgets.QInputDialog.getItem(
                self.view,
                "Select Input Device",
                "Available input devices:",
                device_list,
                0,
                False
            )
            if ok and item:
                # Extract the index from the selected item
                selected_index = int(item.split(":")[0])
                # Find the selected device
                selected_device = next(device for device in input_devices if device['index'] == selected_index)
                return selected_device
            else:
                return None
        except Exception as e:
            print(f"Error accessing input devices: {e}")
            return None

    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for the audio stream. Copies incoming audio data to the buffer.
        """
        if status:
            print(status)
        # Copy audio data to buffer
        self.data = indata[:, 0]

    def smooth(self, data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')


    def update_plot(self):
        """
        Performs FFT on the audio data, computes the outer product, and refreshes the 3D plot.
        """
        # Perform FFT
        windowed_data = self.data * np.hanning(len(self.data))  # Use Hanning window
        fft_data = np.abs(np.fft.rfft(windowed_data))

        # Limit to frequencies up to 20kHz
        fft_data = fft_data[self.freq_indices]
        # Reduce the number of bins
        fft_data = fft_data[self.bin_indices]

        # Normalize FFT data for visualization
        fft_data = fft_data / np.max(fft_data + 1e-6)  # Avoid division by zero

        # Apply non-linear scaling to enhance contrast
        fft_data = np.power(fft_data, 0.25)  # Adjusted exponent from 0.5 to 0.3

        # Apply smoothing to FFT data
        window_size = 10 # Adjust this value as needed
        fft_data_smoothed = self.smooth(fft_data, window_size)

        # Compute the outer product to create a 2D frequency-frequency matrix
        fft_outer = np.outer(fft_data_smoothed, fft_data_smoothed)

        # Update Z data for the surface plot
        amplitude_scaling = 10000  # Adjust this value as needed
        self.Z = fft_outer * amplitude_scaling

        # Ensure Z does not contain NaN or Inf values
        self.Z = np.nan_to_num(self.Z, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify that Z has the correct shape
        if self.Z.shape != (len(self.x), len(self.y)):
            print(f"Shape mismatch: Z shape {self.Z.shape}, expected {(len(self.x), len(self.y))}")
            return  # Skip updating if shapes do not match

        # Update normalization to match the current range of Z
        self.norm = mcolors.Normalize(vmin=np.min(self.Z), vmax=np.max(self.Z))

        # Apply color mapping based on Z values
        colors = self.cmap(self.norm(self.Z))  # RGBA colors, shape (len(x), len(y), 4)
        colors = colors[..., :3]  # Discard alpha channel if not needed

        # Convert colors to float32
        colors = colors.astype(np.float32)

        # Flatten colors to match the expected shape (num_vertices, 3)
        colors = colors.reshape(self.Z.shape[0] * self.Z.shape[1], 3)

        # Update the surface plot's Z data and colors
        try:
            self.surface.setData(z=self.Z, colors=colors)
        except Exception as e:
            print(f"Error updating Z data: {e}")

    def run(self):
        """
        Runs the Qt application and manages the audio stream lifecycle.
        """
        self.app.exec()
        self.stream.stop()
        self.stream.close()

if __name__ == '__main__':
    sa = SpectrumAnalyzer3D()
    sa.run()
