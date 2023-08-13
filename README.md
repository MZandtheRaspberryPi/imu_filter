# imu_filter

![example workflow](https://github.com/MZandtheRaspberryPi/imu_filter/actions/workflows/pipeline.yaml/badge.svg)

Uses dataset from Szczesna et Al. in 2016. Paper [here](https://www.researchgate.net/publication/307076277_Reference_Data_Set_for_Accuracy_Evaluation_of_Orientation_Estimation_Algorithms_for_Inertial_Motion_Capture_Systems).

Szczęsna, A., Skurowski, P., Pruszowski, P., Pęszor, D., Paszkuta, M., & Wojciechowski, K. (2016, September). Reference data set for accuracy evaluation of orientation estimation algorithms for inertial motion capture systems. In International Conference on Computer Vision and Graphics (pp. 509-520). Springer, Cham.

Algo from Zhao and Wang in 2011. Paper [here](https://ieeexplore.ieee.org/document/5999689).

Zhao, He, and Zheyao Wang. “Motion measurement using inertial
sensors, ultrasonic sensors, and magnetometers with extended kalman
filter for data fusion.“ IEEE Sensors Journal 12.5 (2011): 943-953.

This is helpful for more explanation: https://www.nxp.com/docs/en/application-note/AN3461.pdf

And this for details on the magnetometer integration: https://www.mikrocontroller.net/attachment/292888/AN4248.pdf

"1. Static test with empty reference orientation to get noise level in sensor;
2. Slow rotation about sensor X axis (longer arm);
3. Slow rotation about sensor Y axis (shorter arm);
4. Slow rotation about sensor Z axis (perpendicular to arms);
5. Fast rotation about sensor X axis (longer arm);
6. Fast rotation about sensor Y axis (shorter arm);
7. Fast rotation about sensor Z axis (perpendicular to arms);
8. Push (translation) along X sensor axis;
9. Push (translation) along Y sensor axis;
10. Push (translation) along Z sensor axis;
11. Free slow rotation about all three axes."

## Fact Checking Jacobians
Computing the jacobians can be non-trivial. https://octave-online.net/ serves as a good check on your work, though note it doesn't simplify using trigonometric identities.

What are we doing below?
1. create symbolic variables p q r, and load package
2. create symbolic state variables
3. set state vector
4. set state transition matrix
5. take jacobian of state transition matrix with respect to state
```
syms p q r
syms phi theta psi
x = [phi; theta; psi]
f = [phi + p + sin(phi)*tan(theta)*q + cos(phi)*tan(theta)*r;
     theta + cos(phi)*q - sin(phi)*r;
     psi + sin(phi)*q/cos(theta) + cos(phi)*r/cos(theta)]
A = jacobian(f, x)
```

saito
```
syms omega_x omega_y omega_z
syms phi theta psi
syms delta_t
x = [phi; theta; psi]
f = [phi + omega_x * delta_t + sin(phi)*tan(theta)*omega_y * delta_t + cos(phi)*tan(theta)*omega_z*delta_t;
     theta + cos(phi)*omega_y * delta_t - sin(phi)*omega_z*delta_t;
     psi + sin(phi)*omega_y/cos(theta) * delta_t + cos(phi)*omega_z/cos(theta)*delta_t]
A = jacobian(f, x)
```

'''
syms phi theta psi
X_ROT = [[1 0 0;]
         [0 cos(phi) -sin(phi)];
         [0 sin(phi) cos(phi)]]
Y_ROT = [[cos(theta) 0 sin(theta)];
         [0 1 0];
         [-sin(theta) 0 cos(theta)]]
Z_ROT = [[cos(psi) -sin(psi) 0];
         [sin(psi) cos(psi) 0];
         [0 0 1]]
ROT_MAT = (Z_ROT * (Y_ROT * X_ROT))
ROT_MAT_T = transpose(ROT_MAT)
G = [0; 0; 9.81]
H_PRELIM = ROT_MAT_T * G
H = [psi; H_PRELIM(1); H_PRELIM(2); H_PRELIM(3)]
x = [phi; theta; psi]
h = jacobian(H, x)
'''


or for measurement matrix

```
syms p q r
syms phi theta psi
x = [phi; theta; psi]
g = 9.81
f = -g .* [-sin(theta);
          sin(phi) * cos(theta);
          cos(phi) * cos(theta)]
C = jacobian(f, x)
```

## Points of comparison
[doing ekf for pos and orientation](https://robomechjournal.springeropen.com/articles/10.1186/s40648-020-00185-y)

## Doing this in real life
Make sure that axis match expectations for accelerometer, gyroscope, and magnetometer.
Can check values when stationary and ensure -1g and 1g.

# coordinate system imu 
https://epan-utbm.github.io/utbm_robocar_dataset/docs/MTi-28A53G25.pdf

## Docker

From root of git repo:
```
./build_docker_image.sh
docker login
docker tag sha256:bc9a68bb410cd35bd600cd3e18c84357cccd9cb09f145fa3dc4c709858df55f2 mzandtheraspberrypi/imu-filter-ubuntu22-arm64:2023-08-11
docker push mzandtheraspberrypi/imu-filter-ubuntu22-arm64:2023-08-11
```
