import numpy as np
import time
import cv2
import csv
import os
import collections
import datetime
import math

from object_detection.utils.visualization_utils import STANDARD_COLORS

PERSON_ON_DOOR_THRESHOLD = 0.4  # Amount of overlapping area
CSV_FILE = 'records.csv'
NON_DETECTION_REMOVAL_TIME = 120    # Seconds

coloring_index_at = 0

def get_screen_size():
	return (540, 320)

def get_door_coods():
	door_point1_at = (0.7, 0.6, 0.8, 1.0)
	door_points_at = [door_point1_at]
	return door_points_at

class ObjectCood:
	def __init__(self, x_min, y_min, x_max, y_max):
		self.update_cood(x_min, y_min, x_max, y_max)
		
	def update_cood(self, x_min, y_min, x_max, y_max):
		self.x_min = x_min
		self.y_min = y_min
		self.x_max = x_max
		self.y_max = y_max

		self.center_x = (x_max - x_min) / 2
		self.center_y = (y_max - y_min) / 2

	def get_distance_from(self, x_min, y_min, x_max, y_max):
		center_x = (x_max - x_min) / 2
		center_y = (y_max - y_min) / 2
		screen_size = get_screen_size()
		distance = math.sqrt((((center_x - self.center_x) * screen_size[0]) ** 2) +
						(((center_y - self.center_y) * screen_size[1]) ** 2))
		return distance

	def get_area_in_map(self):
		screen_size = get_screen_size()
		area = ((self.x_max - self.x_min) * screen_size[0]) * ((self.y_max - self.y_min) * screen_size[1]) 
		return area


class PersonData:
	def __init__(self, x_min, y_min, x_max, y_max, score):
		self.last_tracked = int(time.time())
		self.coods = ObjectCood(x_min, y_min, x_max, y_max)
		self.color_track = None # Future
		self.score = score
		self.is_person_on_door = False

	def update_location(self, x_min, y_min, x_max, y_max, score):
		self.coods.update_cood(x_min, y_min, x_max, y_max)
		self.score = score
		self.update_latest_time()
		self.update_is_person_on_door()

	def update_latest_time(self):
		self.last_tracked = int(time.time())

	def get_distance_from(self, x_min, y_min, x_max, y_max):
		return self.coods.get_distance_from(x_min, y_min, x_max, y_max)

	def calculate_area_of_intersection(self, box_x_min, box_y_min, box_x_max, box_y_max):
		box1 = (self.coods.x_min, self.coods.y_min, self.coods.x_max, self.coods.y_max)
		box2 = (box_x_min, box_y_min, box_x_max, box_y_max)
		dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
		dy = min(box1[3], box2[3]) - max(box1[1], box2[1])

		screen_size = get_screen_size()
		dx *= screen_size[0]
		dy *= screen_size[1]
		area_of_intersection = dx * dy
		if area_of_intersection < 0:
			area_of_intersection = 0
		return area_of_intersection

	def update_is_person_on_door(self):
		door_coods = get_door_coods()
		area_in_map = self.coods.get_area_in_map()
		threshold_area = PERSON_ON_DOOR_THRESHOLD * area_in_map
		self.is_person_on_door = False
		for door_cood in door_coods:
			intersection_area = self.calculate_area_of_intersection(door_cood[0], door_cood[1], door_cood[2],
				door_cood[3])
			if intersection_area >= threshold_area:
				self.is_person_on_door = True
			break

	def set_boundary_color(self):
		if coloring_index_at >= len(STANDARD_COLORS):
			coloring_index_at = 0
		self.color_track = STANDARD_COLORS[coloring_index_at]
		coloring_index_at += 1

class PeopleTracker:

	def __init__(self, min_score_thresh):
		self.min_score_thresh = min_score_thresh
		self.person_array = []
		self.current_boxes_len = 0
		self.door_locations = self.get_door_locations()
	
	def get_door_locations(self):
		screen_size = get_screen_size()
		door_points_at = get_door_coods()

		door_coods_at = []
		for door_point in door_points_at:
			door_cood_at = ((int(screen_size[0] * door_point[0]), int(screen_size[1] * door_point[1])), 
				(int(screen_size[0] * door_point[2]), int(screen_size[1] * door_point[3])))
			door_coods_at.append(door_cood_at)
		return door_coods_at

	def plot_doors(self, screen_np):
		for door_cood_at in self.door_locations:
			cv2.rectangle(screen_np, door_cood_at[0], door_cood_at[1], (0, 0, 255), 2)

	def remove_junk_data(self):
		deleting_indexes = []
		current_time = int(time.time())
		for index, person in enumerate(self.person_array):
			if current_time - person.last_tracked > NON_DETECTION_REMOVAL_TIME:
				deleting_indexes.append(index)
		for index in sorted(deleting_indexes, reverse = True):
			del self.person_array[index]

	def save_recorded_data_in_file(self):
		if not os.path.isfile(CSV_FILE):
			with open(CSV_FILE, 'w') as csv_file:
				header_row = ['ReadableTime', 'TimeStamp', 'BoxesPlot', 'AsPerCalculation']
				writer = csv.writer(csv_file)
				writer.writerow(header_row)

		current_timestamp = int(time.time())
		readable_time = datetime.datetime.fromtimestamp(current_timestamp).strftime('%Y-%m-%d %H:%M:%S')
		data_to_write = [readable_time, str(current_timestamp), str(self.current_boxes_len), str(len(self.person_array))]
		with open(CSV_FILE, 'a') as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(data_to_write)


	def make_initial_analysis(self, boxes, scores):
		for box, score in zip(boxes, scores):
			if score < self.min_score_thresh:     # Data will be sent in descending order
				break

			person = PersonData(box[0], box[1], box[2], box[3], score)
			self.person_array.append(person)

	def update_person_tracker(self, boxes, scores):
		reqd_boxes, reqd_scores = [], []
		for box, score in zip(boxes, scores):
			if score < self.min_score_thresh:
				break

			reqd_boxes.append(box)
			reqd_scores.append(score)

		if len(reqd_boxes) == 0:
			return reqd_boxes

		box_distance_array = []
		for box in reqd_boxes:
			box_distances = [person.get_distance_from(box[0], box[1], box[2], box[3]) for person in self.person_array]
			box_distance_array.append(box_distances)
		min_box_distance_index = np.argmin(box_distance_array, axis = 1)
		#print('Box distance array: {}'.format(box_distance_array))
		#print('Min box distance index: {}'.format(min_box_distance_index))

		added_boxes = []
		if len(self.person_array) > 0:
			person_distance_array = []
			area_of_intersection_array = []
			for person in self.person_array:
				person_distances = [person.get_distance_from(box[0], box[1], box[2], box[3]) for box in reqd_boxes]
				person_distance_array.append(person_distances)

				area_of_intersections = [person.calculate_area_of_intersection(box[0], box[1], box[2], box[3]) for box in reqd_boxes]
				area_of_intersection_array.append(area_of_intersections)

			min_person_distance_index = np.argmin(person_distance_array, axis = 1)
			max_area_of_intersection_index = np.argmax(area_of_intersection_array, axis = 1)

			competiting_array = [-1] * len(self.person_array)
			# First iterate through person_array and estimate the possible nearest box for each rectangle.
			for index, person in enumerate(self.person_array):
				nearest_box_by_person = min_person_distance_index[index]
				nearest_person_by_box = min_box_distance_index[nearest_box_by_person] 
				max_overlapping_by_person = max_area_of_intersection_index[index]

				if nearest_person_by_box == index and (nearest_box_by_person == max_overlapping_by_person or 
					area_of_intersection_array[index][nearest_box_by_person] > 0):
					competiting_array[index] = nearest_box_by_person
			
			repeat_counter = collections.Counter(competiting_array)
			repeat_counter_keys = repeat_counter.keys()
			# Break the ties by distances
			for key in repeat_counter_keys:
				repeated_times = repeat_counter[key]
				if key == -1 or repeated_times == 1:
					continue

				# Give highest preference to nearest person by box.
				box_distances = box_distance_array[key]
				person_indices = [i[0] for i in sorted(enumerate(box_distances), key=lambda x:x[1])]
				is_tie_broken = False
				for person_index in person_indices:
					if competiting_array[person_index] == -1:
						#Make all the other elements as -1 in competiting array
						for c_index in range(len(competiting_array)):
							if c_index != person_index and competiting_array[c_index] == key:
								competiting_array[c_index] = -1
						is_tie_broken = True
						break
				# If tie is not broken, make all the elements as -1
				if not is_tie_broken:
					for c_index in range(len(competiting_array)):
						if competiting_array[c_index] == key:
							competiting_array[c_index] = -1

			repeat_counter = collections.Counter(competiting_array)
			repeat_counter_keys = repeat_counter.keys()
			# Now, inside competiting array, if elements are getting repeated, need to break the tie
			deleting_indexes = []
			for index, person in enumerate(self.person_array):
				competiting_box_index = competiting_array[index]
				if competiting_box_index == -1:
					nearest_box_by_person = min_person_distance_index[index]
					nearest_person_by_box = min_box_distance_index[nearest_box_by_person] 
					if person.is_person_on_door:
						# Treat as person has left
						deleting_indexes.append(index)
					elif nearest_person_by_box == index and repeat_counter[nearest_box_by_person] == 0:
						reqd_box = reqd_boxes[nearest_box_by_person]
						person.update_location(reqd_box[0], reqd_box[1], reqd_box[2], reqd_box[3], reqd_scores[nearest_box_by_person])
						added_boxes.append(nearest_box_by_person)
				elif repeat_counter[competiting_box_index] == 1:
					reqd_box = reqd_boxes[competiting_box_index]
					person.update_location(reqd_box[0], reqd_box[1], reqd_box[2], reqd_box[3], reqd_scores[competiting_box_index])
					added_boxes.append(competiting_box_index)
				else:
					# This condition shouldn't be hit.
					print('Competiting box > 1 condition hit')
					pass
			
			# Next take care of all the deleting persons.
			for index in sorted(deleting_indexes, reverse=True):
				del self.person_array[index]

		# Next, add people who are not present.
		for index, (box, score) in enumerate(zip(reqd_boxes, reqd_scores)):
			if index not in added_boxes:
				person = PersonData(box[0], box[1], box[2], box[3], score)
				self.person_array.append(person)

		self.current_boxes_len = len(reqd_boxes)
		return reqd_boxes

	def draw_boundary_boxes(self, reqd_boxes):
		pass
				
