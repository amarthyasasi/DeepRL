from flask import Flask, render_template_string, Response

import airsim
import cv2
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()

CAMERA_NAME = '0'
IMAGE_TYPE = airsim.ImageType.Scene
DECODE_EXTENSION = '.jpg'

#camera_pose = airsim.Pose(airsim.Vector3r(-6, 0, -2), airsim.to_quaternion(0, 0, 0))  #PRY in radians
#print(drone_pose, camera_pose)

def frame_generator():
    while (True):
        drone_pose = client.simGetVehiclePose()
        #drone_pose.position.x_val = drone_pose.position.x_val - 6
        k = client.getMultirotorState().kinematics_estimated.position
        #drone_pose.position.z_val = drone_pose.position.z_val - 2
        #drone_pose.position.z_val = drone_pose.position.y_val - 6
        drone_pose.position = k
        #drone_pose.position.z_val = drone_pose.position.z_val - 6
        drone_pose.orientation.z_val = 0
        drone_pose.orientation.x_val = 0
        drone_pose.orientation.y_val = -0.3
        drone_pose.orientation.w_val = 0.9
        #print(drone_pose)
        client.simSetCameraPose(0, drone_pose);
        response_image = client.simGetImage(CAMERA_NAME, IMAGE_TYPE)
        np_response_image = np.asarray(bytearray(response_image), dtype="uint8")
        decoded_frame = cv2.imdecode(np_response_image, cv2.IMREAD_COLOR)
        #print(decoded_frame.shape, decoded_frame.max())
        decoded_frame = cv2.resize(decoded_frame, (1080, 720), interpolation = cv2.INTER_AREA)
        ret, encoded_jpeg = cv2.imencode(DECODE_EXTENSION, decoded_frame)
        frame = encoded_jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(
        """
            <html>
            <head>
                <title>AirSim Streamer</title>
            </head>
            <body>
                <h1>AirSim Streamer</h1>
                <hr />
                Please use the following link: <a href="/video_feed">http://localhost:5000/video_feed</a>
            </body>
            </html>
        """
        )

@app.route('/video_feed')
def video_feed():
    return Response(
            frame_generator(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)