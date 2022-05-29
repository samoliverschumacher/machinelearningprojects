A project for a "Microsleep detector", which I've blogged about in detail: https://samoliverschumacher.github.io/website/post/eeg_eda/

> *Produce a detector that distinguisihes when a car’s driver has eyes closed. This detector will connect to an autopilot system for the car temporarily until driver regains control, otherwise safely brings the car to a halt on the side of the road. Detector will have access to EEG signals connected to drivers head.*

## Running the project;

Each of the main files are suffixed with `_script`. all others are helpers / implementations;
1. First load all files into one folder
2. Download CVX, and install it (convex optimisation program for MATLAB)
3. Run files in order: EDA, Modelling, Validation.

Each script saves data, which the next script will have to call on.
The end of "Validation" will not work, as it requires a few saved workspaces from running the LSTM model with various configurations.

