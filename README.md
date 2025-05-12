# 3D-Computer-Vision-And-Robotics
Calibrating the robot's environment using camera calibration with DLT, followed by tasks such as cube picking and transporting it to a designated location


<!DOCTYPE html>
<html lang="en">

<body>

<h1>3D-Computer-Vision-And-Robotics</h1>

<p>This project demonstrates a robot's ability to understand and interact with its 3D environment using computer vision. The robot is calibrated using a camera and Direct Linear Transform (DLT) technique, enabling it to locate colored cubes and transport them to specific targets.</p>

<h2>📷 1. Camera Calibration (DLT)</h2>
<p>To convert 2D image coordinates into 3D world coordinates, the camera was calibrated using the <strong>Direct Linear Transform (DLT)</strong>. A known set of 3D-2D correspondences was used to estimate the projection matrix <code>P</code>, allowing us to reconstruct 3D coordinates from a single image.</p>

<h3>🔧 Key Steps</h3>
<ol>
    <li>Capture known 3D object points and their 2D image projections.</li>
    <li>Solve for the 3x4 projection matrix <code>P</code> using least squares.</li>
    <li>Use <code>P</code> to map future 2D detections into 3D world space.</li>
</ol>

<h2>📍 2. Localization of Robot and Objects</h2>
<p>Using the camera’s projection matrix, we estimate the 3D positions of all important elements:</p>
<ul>
    <li>Robot base and arm</li>
    <li>Colored cubes (<code>r_cube</code>, <code>g_cube</code>, <code>b_cube</code>)</li>
    <li>Target locations (<code>*_target</code>)</li>
</ul>

<h3>🔲 2D View of Localized Points (Z Flattened)</h3>
<img src="newplot.png" alt="2D Plot of Localized Points">

<h3>🧭 3D Spatial Layout</h3>
<p>The following 3D plot visualizes the spatial relationship between robot components and objects:</p>
<img src="3dplot.png" alt="3D Point Plot">

<h2>📐 3. Distance and Angle Computation</h2>
<p>Once all objects are localized, we compute:</p>
<ul>
    <li><strong>Euclidean Distance</strong> from the robot's arm to each cube.</li>
    <li><strong>Orientation Angle</strong> required for the arm to align with the cube’s position.</li>
</ul>


</body>
</html>

