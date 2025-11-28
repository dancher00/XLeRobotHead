# Palm State Angle Calculation Explanation

## Overview
This document explains how we calculate roll, pitch, and yaw angles from MediaPipe hand landmarks.

## Coordinate Systems

### MediaPipe Coordinate System (Input)
- **Origin**: Top-left corner of image
- **X axis**: 0.0 (left) → 1.0 (right) - horizontal in image
- **Y axis**: 0.0 (top) → 1.0 (bottom) - vertical in image
- **Z axis**: Relative depth (negative = closer to camera, positive = farther)

### Target Coordinate System (Output)
- **X axis**: Forward (positive = toward camera)
- **Y axis**: Left (positive = to the left)
- **Z axis**: Up (positive = upward)

### Transformation
```
X_target = -Z_mediapipe  (MediaPipe forward → our forward)
Y_target = -X_mediapipe  (MediaPipe right → our left)
Z_target = -Y_mediapipe  (MediaPipe down → our up)
```

## Angle Calculation Process

### Step 1: Extract Key Landmarks
- **Wrist** (landmark 0): Base of hand
- **Index MCP** (landmark 5): Index finger base
- **Middle MCP** (landmark 9): Middle finger base
- **Pinky MCP** (landmark 17): Pinky finger base

### Step 2: Calculate Palm Plane
1. Create two vectors in the palm plane:
   - `v1 = index_mcp - wrist` (wrist to index)
   - `v2 = pinky_mcp - wrist` (wrist to pinky)

2. Calculate palm normal (perpendicular to palm):
   - `normal = v1 × v2` (cross product)
   - Normalize: `normal = normal / ||normal||`

### Step 3: Build Coordinate Frame
We construct three orthogonal vectors:

1. **Forward (X)**: Direction palm is facing
   - `forward_mp = -normal`
   - When palm faces camera, normal points toward camera, so -normal gives forward direction

2. **Left (Y)**: Left side of hand
   - `left_mp = (pinky_mcp - wrist)` projected onto palm plane
   - Projected to be perpendicular to forward
   - Normalized

3. **Up (Z)**: Up along the hand
   - `up_mp = forward_mp × left_mp` (cross product)
   - This ensures right-handed coordinate system
   - Normalized

### Step 4: Transform to Target Coordinate System
Transform each vector from MediaPipe space to target space:
```
forward_new = [-forward_mp[2], -forward_mp[0], -forward_mp[1]]
left_new = [-left_mp[2], -left_mp[0], -left_mp[1]]
up_new = [-up_mp[2], -up_mp[0], -up_mp[1]]
```

### Step 5: Build Rotation Matrix
Create rotation matrix R where columns are [forward, left, up]:
```
R = [forward_new | left_new | up_new]
```

This matrix represents the orientation of the palm in the target coordinate system.

### Step 6: Extract Euler Angles (ZYX Convention)
From rotation matrix R, extract angles:

- **Yaw** (rotation around Z/up axis):
  ```
  yaw = atan2(R[1,0], R[0,0])
  ```
  - Represents horizontal rotation (which direction palm is pointing)

- **Pitch** (rotation around Y/left axis):
  ```
  pitch = asin(-R[2,0])
  ```
  - Represents vertical tilt (up/down)

- **Roll** (rotation around X/forward axis):
  ```
  roll = atan2(R[2,1], R[2,2])
  ```
  - Represents lateral tilt (left/right)

### Step 7: Convert to Quaternion
Convert Euler angles to quaternion [w, x, y, z] using standard formulas.

## Verification

The angles should behave as follows:
- **Roll**: When you tilt palm left/right, roll changes
- **Pitch**: When you tilt palm up/down, pitch changes  
- **Yaw**: When you rotate palm horizontally, yaw changes

## Notes

1. **Depth (Z)**: MediaPipe's Z is relative, not absolute. It's estimated from hand size and pose geometry.

2. **Coordinate Frame**: The frame is constructed from palm geometry, ensuring it represents the actual palm orientation.

3. **Euler Angle Convention**: ZYX (yaw-pitch-roll) is used, which is standard for aerospace/robotics applications.

4. **Gimbal Lock**: At extreme angles (pitch = ±90°), there may be gimbal lock issues. The quaternion representation avoids this.

