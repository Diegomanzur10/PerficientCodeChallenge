import shutil
import numpy as np
import os
import tarfile
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
import math
import itertools
import matplotlib.patches as patches
from random import randrange
import cv2
import matplotlib
from matplotlib import pyplot as plt
plt.ioff()
matplotlib.use('Agg')
import moviepy.editor as mp
import wget
import pandas as pd


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


FOLDER_INPUT = '/tf/video/'

PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/{}.tar.gz'.format(PRETRAINED_MODEL_NAME)
paths = {
    'DOWNLOAD_MODEL_PATH': os.path.join(os.getcwd(), PRETRAINED_MODEL_NAME + '.tar.gz'),
    'PASTE_MODEL_PATH': os.path.join(os.path.join(os.getcwd(), "preTrainedModels")), 
 }

# list to store files
res = []
# Iterate directory
for path in os.listdir(FOLDER_INPUT):
    # check if current path is a file
    if os.path.isfile(os.path.join(FOLDER_INPUT, path)):
        res.append(os.path.join(FOLDER_INPUT, path))

FILE_INPUT = res[0]
FILE_RESIZED = os.path.join(os.getcwd(), "resize_video.mp4")

NAME_OUTPUT_VIDEO = input("Enter the name of outputvideo (without format accronim): ")
FILE_OUTPUT = '/tf/video/{}.mp4'.format(NAME_OUTPUT_VIDEO)

LENGTH_TO_CHECK = int(input("Enter the distance to check in meters: "))

# Resizing the video

clip = mp.VideoFileClip(FILE_INPUT, target_resolution = (450, 800), resize_algorithm = "gauss")
# initial_fps_rate = clip.fps
final_fps_rate = 2

if final_fps_rate < clip.fps:
    clip = clip.set_fps(final_fps_rate)
    clip = clip.subclip(0, clip.duration / final_fps_rate)

clip.write_videofile(FILE_RESIZED, codec = "mpeg4", audio = False, preset = "ultrafast", threads = 2)
clip.close()


##### Create folders

# preTrainedModels folder
if not os.path.exists(paths['PASTE_MODEL_PATH']):
    os.makedirs(paths['PASTE_MODEL_PATH'])

##################################################  Helpers #############################################################
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def filter_boxes(min_score, boxes, scores, classes, categories):
  """Return boxes with a confidence >= `min_score`"""
  n = len(classes)
  idxs = []
  for i in range(n):
      if classes[i] in categories and scores[i] >= min_score:
          idxs.append(i)
  
  filtered_boxes = boxes[idxs, ...]
  filtered_scores = scores[idxs, ...]
  filtered_classes = classes[idxs, ...]
  return filtered_boxes, filtered_scores, filtered_classes

def calculate_coord(bbox, width, height):
    xmin = bbox[1] * width
    ymin = bbox[0] * height
    xmax = bbox[3] * width
    ymax = bbox[2] * height

    return [xmin, ymin, xmax - xmin, ymax - ymin]

def calculate_centr(coord):
  return (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))

def calculate_centr_distances(centroid_1, centroid_2):
  return  math.sqrt((centroid_2[0]-centroid_1[0])**2 + (centroid_2[1]-centroid_1[1])**2)

def calculate_perm(centroids):
  permutations = []
  for current_permutation in itertools.permutations(centroids, 2):
    if current_permutation[::-1] not in permutations:
      permutations.append(current_permutation)
  return permutations

def midpoint(p1, p2):
    return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

def calculate_slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m



# Download the pretrained model
url=PRETRAINED_MODEL_URL
wget.download(url)
if not os.path.exists(os.path.join(paths['PASTE_MODEL_PATH'], PRETRAINED_MODEL_NAME + ".tar.gz")):
    shutil.move(paths['DOWNLOAD_MODEL_PATH'], os.path.join(paths['PASTE_MODEL_PATH'], PRETRAINED_MODEL_NAME + ".tar.gz"))
if os.path.exists("demofile.txt"):
    os.remove(paths['DOWNLOAD_MODEL_PATH'])
file = tarfile.open(os.path.join(paths['PASTE_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))
file.extractall(paths['PASTE_MODEL_PATH'])
file.close()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = os.path.join(paths['PASTE_MODEL_PATH'], PRETRAINED_MODEL_NAME, "frozen_inference_graph.pb")


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = r'/tf/mscoco_label_map.pbtxt'


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef() 
  with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Playing video from file
cap = cv2.VideoCapture(FILE_RESIZED)

cap.set(cv2.CAP_PROP_FPS, int(final_fps_rate))


# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
width = int(cap.get(3))
height = int(cap.get(4))


i = 0
new = True
list_less_meter = []
list_midpoing_less_meter_x = []
list_midpoing_less_meter_y = []

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        i = 0
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Correct color
                frame = gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                
                # Filter boxes
                confidence_cutoff = 0.5
                boxes, scores, classes = filter_boxes(confidence_cutoff, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), [1])

                # Calculate normalized coordinates for boxes
                centroids = []
                coordinates = []
                for box in boxes:
                    coord = calculate_coord(box, width, height)
                    centr = calculate_centr(coord)
                    centroids.append(centr)
                    coordinates.append(coord)

                # Pixel per meters
                average_px_meter = (width) / 6

                permutations = calculate_perm(centroids)

                # Display boxes and centroids
                fig, ax = plt.subplots(figsize = (20,12), dpi = 90, frameon=False)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for coord, centr in zip(coordinates, centroids):
                    ax.add_patch(patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], linewidth=2, edgecolor='g', facecolor='none', zorder=10))
                    ax.add_patch(patches.Circle((centr[0], centr[1]), 3, color='green', zorder=20))

                # Display lines between centroids
                for perm in permutations:
                    dist = calculate_centr_distances(perm[0], perm[1])
                    dist_m = dist/average_px_meter
                    
                    x1 = perm[0][0]
                    y1 = perm[0][1]
                    x2 = perm[1][0]
                    y2 = perm[1][1]

                    # Calculate middle point
                    middle = midpoint(perm[0], perm[1])

                    # Calculate slope
                    slope = calculate_slope(x1, y1, x2, y2)
                    dy = math.sqrt(3**2/(slope**2+1))
                    dx = -slope*dy

                    # Set random location
                    if randrange(10) % 2== 0:
                        Dx = middle[0] - dx*10
                        Dy = middle[1] - dy*10
                    else:
                        Dx = middle[0] + dx*10
                        Dy = middle[1] + dy*10

                    if dist_m < LENGTH_TO_CHECK:
                        #Save te distance when we hava an incident, as well as the centroid of this incident
                        list_less_meter.append(dist_m)
                        list_midpoing_less_meter_x.append(middle[0])
                        list_midpoing_less_meter_y.append(middle[1])

                        ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='white', xytext=(Dx, Dy), fontsize=10, arrowprops=dict(arrowstyle='->', lw=1.5, color='yellow'), bbox=dict(facecolor='red', edgecolor='white', boxstyle='round', pad=0.2), zorder=35)
                        ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]), linewidth=2, color='yellow', zorder=15)

                    # else:
                    #     ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='green', xytext=(Dx, Dy), fontsize=8, arrowprops=dict(arrowstyle='->', lw=1.5, color='skyblue'), bbox=dict(facecolor='g', edgecolor='white', boxstyle='round', pad=0.2), zorder=35)
                    #     ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]), linewidth=2, color='skyblue', zorder=15)

                
                ax.imshow(frame, interpolation='nearest')
                
                # This allows you to save each frame in a folder to process each frame and then merge the entire video
                #fig.savefig("tf/test_images/TEST_{}.png".format(i))
                # i += 1

                # Convert figure to numpy
                fig.canvas.draw()

                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                img = np.array(fig.canvas.get_renderer()._renderer)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

                if new:
                    print("Define out")
                    out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc(*'MP4V'), final_fps_rate, (img.shape[1], img.shape[0]))
                    new = False
                    if os.path.exists(FILE_OUTPUT):
                        print("Saving video")
                    else:
                        print("No Saving Video")

                out.write(img)
            else:
                break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

# Save summary
df_summary = pd.DataFrame({"distance_incident": list_less_meter,
                            "coordinate_x_incident": list_midpoing_less_meter_x,
                            "coordinate_y_incident": list_midpoing_less_meter_y})

df_summary.to_csv(os.path.join(FOLDER_INPUT, "summary.txt"))