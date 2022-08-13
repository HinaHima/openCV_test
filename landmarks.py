import cv2
import mediapipe as mp

# Face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Image
image = cv2.imread("check.jpg")
height, width, _ = image.shape
print(f"Height = {height}, widht = {width}")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Facial landmarks
result = face_mesh.process(rgb_image)

for facial_landmarks in result.multi_face_landmarks:
    print(f"Face landmarks: {facial_landmarks}")
    #mp_drawing.draw_landmarks(
        #image=image,
        #landmark_list=facial_landmarks,
        #connections=mp_face_mesh.FACEMESH_TESSELATION,
        #landmark_drawing_spec=None,
        #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    #)

    vertices = []

    for i in range(0, 468):
        pt1 = facial_landmarks.landmark[i]
        x = pt1.x
        y = pt1.y
        z = pt1.z
        vertices.append((x, y, z),)

    print(vertices)

        #cv2.circle(image, (x, y), 2, (0, 0, 250), -1)

    #cv2.imwrite('meshed_image' + '.png', image )

#cv2.imshow("Image", image)
#cv2.waitKey(0)