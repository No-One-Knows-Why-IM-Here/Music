The main idea of this project is to analyze sound frequencies and display 
them in a 3D graph. The x and y axes represent frequency, 
while the z axis represents amplitude. The range of possible frequencies is 
from 0 to 20 kHz (since we can't hear more than that! :p).

There are some important features in the code that are easy to change. 
Want to change the resolution? Adjust `self.desired_bins = n`. 
Want to change the graph update time? Tweak `self.timer.start(n)`. 
Want to change how smooth the graph is? Modify `window_size = n`.
and so on
