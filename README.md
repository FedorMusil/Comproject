# Shallow Water Equation Simulation

## This project is a simple demonstration of a shallow water equation simulation around a recreation of Australia.

### What this application does
We used a very basic shallow water equation with finite differences to simulate how water flows. With simple boundary conditions, we let the water flow back once it hits a wall, including the borders of the plot as well as that of any polygons placed in the plot. Using the winding number algorithm and a simple coordinates system, we can plot essentially any shape into the simulation and determine if any given coordinate is in an island. If so, we make any water waves that hit said island reflect back into the ocean.

We went with this approach for multiple reasons, but the most prominent one is that finite difference equations are relatively lightweight while still providing good results. Initially, we wanted to go for a much grander scale, possibly scaling up to simulate the whole world, but we quickly realised this was not computationally feasible within the allotted time.

### How to use this project
The entire simulation and plotting takes place within SWE.py. It creates at most 2 videos of a specified length using commandline arguments.

Running `python SWE.py -mv` will produce a 20 second long video of the shallow water equation simulation named "velocity_field.mp4".

#### Multiprocessing
This program does support the use of multiprocessing to speed up the video creation process. This method makes use of the tool ffmpeg, splitting up the video rendering process into segments before stitching them together. Ffmpeg can be downloaded [here](https://www.ffmpeg.org/download.html). In order for this program to work, the ffmpeg executable has to be added to your operating system's PATH. Once ffmpeg has been properly installed, running `python SWE.py -mp -mv` will, by default, create 4 processes to produce 4 segments of the video, before stitching the resulting videos together into 1 video.

Running `python SWE.py -h` will give a full list of accepted arguments.

#### Polygons
This project does support polygonal shapes. Currently, only the island of Australia is provided in the islands.py file, as a long list of coordinates. These values are currently hardcoded into the project itself, but the code in SWE.py is robust enough to support multiple polygons, it will just require some slight altercations to make this more modular and was out of the scope of this project. Note that this method of allowing polygons does not provide a perfect 1:1 of the polygonal shape in the calculation. To show this, this gif shows the outline of Australia in red, with the actual area affected by said polygons drawn in green.

![Image time](/velocity_field.gif "Preview of our shallow water equation simulation.")

### Credits
This project was made by 4 students: Fedor Musil, Sam Baldinger, Kim Roks and Tim Roepke.

We made use of [this shallow water equation simulator by jostbr](https://github.com/jostbr/shallow-water) as inspiration for both the shallow water equations and the simulation code itself.

### License
This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).