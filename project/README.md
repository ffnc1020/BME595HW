# Title
Backprop SVSF for Attitude Estimation under Large Instantaneous Vibration
## Team members
Fan Fei (ffnc1020), Yilun Yang (YilunYang)
## Goals
1. Introduction
Unlike conventional aerial vehicles of fixed or rotary wings, onboard attitude estimation of insect or hummingbird scale Flapping Wing Micro Aerial Vehicles (FWMAVs) is very challenging due to the severe oscillations (approximately 10 times of gravity on our platform) induced by high-frequency wing flapping.
We propose to use neural network to create a discriminative model for state estimation (roll pitch yaw angle) from inertia measurement unit (IMU).
2. Data acquisition
In order to generalize different vehicle platforms, sensors and operating points, and enables the network to learn the intrinsic dynamics of the sensor and vehicle dynamics, large amount of data are needed. With the proptotype platform we developed in our lab, we plan to acquire raw sensor measurement data from 4 different IMU sensors, under different operating conditions (hovering, maneuvering).
3. Proposed Model
We propose to use a hybrid model structure combining LSTM and SVSF filter which utilize the rich temproal information from LSTM and the convergence properities from SVSF.
The model take 9 timeseries data from gyro, accelerometer and magnetometer and output the quaternion of the orientatin, then convert to three attitude angles.
## Challenges
1. Experimental work that involves hardware is always difficult. Currently we do not have the data acquisition setup to get ground truth from Vicon cameras. Since this method is data driven, we need to conduct enought test flights to acquire sufficient data.
2. The SVSF filter has non-differentiable components which we need to come up with tricks to make it work with back propagation.
3. Since we have zero experience with recurrent neural network, the training and tuning could be challenging.
## Initial Results
We use a “vanilla” RNN network with three hidden layers and each layer has 10 neurons. The initial result is plotted as follows. The root mean square error (RMSE) of each axis is 4.1192, 4.5366 and 4.7619 degrees for roll, pitch and yaw axis respectively, which is slightly higher than the state-of-the-art method which directly leveraging the physics and sensor characteristics.

