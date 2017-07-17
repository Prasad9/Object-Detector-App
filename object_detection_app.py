import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from PeopleData import PeopleTracker

RECORD_DATA_INTERVAL_TIME = 2       # Seconds

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

min_score_thresh = 0.5

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

people_tracker = PeopleTracker(min_score_thresh = min_score_thresh)

def map_reqd_class(search_classes):
	reqd_class_array = []
	for key, value_dict in category_index.items():
		if value_dict['name'] in search_classes:
			reqd_class_array.append(value_dict['id'])
	return reqd_class_array
reqd_class_array = map_reqd_class(['person'])

def detect_objects(image_np, sess, detection_graph):
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

	# Each box represents a part of the image where a particular object was detected.
	boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	scores = detection_graph.get_tensor_by_name('detection_scores:0')
	classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	# Actual detection.
	(boxes, scores, classes, num_detections) = sess.run(
		[boxes, scores, classes, num_detections],
		feed_dict={image_tensor: image_np_expanded})
	
	marking_boxes = []
	marking_scores = []
	marking_classes = []
	for box, score, class_type in zip(boxes[0], scores[0], classes[0]):
		if class_type in reqd_class_array:
			marking_boxes.append(box)
			marking_scores.append(score)
			marking_classes.append(class_type)

	people_tracker.plot_doors(image_np)
	#print('Boxes: {}'.format(marking_boxes))
	#print('Scores: {}'.format(marking_scores))
	#print('Classes: {}'.format(marking_classes))
	return marking_boxes, marking_classes, marking_scores

def plot_objects(image_np, boxes, classes, scores):
	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(
		image_np,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates = True,
		line_thickness = 8,
		min_score_thresh = min_score_thresh)
	
	return image_np

def worker(input_q, output_q):
	# Load a (frozen) Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	sess = tf.Session(graph=detection_graph)

	fps = FPS().start()
	
	initial_frame = input_q.get()
	boxes, classes, scores = detect_objects(initial_frame, sess, detection_graph)
	people_tracker.make_initial_analysis(boxes, scores)	
	output_q.put(plot_objects(initial_frame, boxes, classes, scores))

	assumed_fps = 15
	record_counter_at = 0
	record_counter_limit = assumed_fps * RECORD_DATA_INTERVAL_TIME

	while True:
		fps.update()
		frame = input_q.get()

		boxes, classes, scores = detect_objects(frame, sess, detection_graph)
		
		_ = people_tracker.update_person_tracker(boxes, scores)

		record_counter_at += 1
		if record_counter_at >= record_counter_limit:
			people_tracker.save_recorded_data_in_file()
			record_counter_at = 0

		people_tracker.remove_junk_data()
		
		output_q.put(plot_objects(frame, boxes, classes, scores))

	fps.stop()
	sess.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-src', '--source', dest='video_source', type=int,
		        default=0, help='Device index of the camera.')
	parser.add_argument('-wd', '--width', dest='width', type=int,
		        default=480, help='Width of the frames in the video stream.')
	parser.add_argument('-ht', '--height', dest='height', type=int,
		        default=360, help='Height of the frames in the video stream.')
	parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
		        default=2, help='Number of workers.')
	parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
		        default=5, help='Size of the queue.')
	args = parser.parse_args()

	logger = multiprocessing.log_to_stderr()
	logger.setLevel(multiprocessing.SUBDEBUG)

	input_q = Queue(maxsize=args.queue_size)
	output_q = Queue(maxsize=args.queue_size)

	pool = Pool(args.num_workers, worker, (input_q, output_q))

	video_capture = WebcamVideoStream(src=args.video_source,
		                      width=args.width,
		                      height=args.height).start()

	fps = FPS().start()

	while True:  # fps._numFrames < 120
		frame = video_capture.read()

		input_q.put(frame)

		t = time.time()
		display_frame = output_q.get()
		cv2.imshow('Video', display_frame)
		fps.update()

		#print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	fps.stop()
	print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
	print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

	pool.terminate()
	video_capture.stop()
	cv2.destroyAllWindows()
