'''
   Copyright 2017 Larry Chen

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import numpy as np

def extract_bounding_boxes(scene, names):
  objs = scene['objects']
  rotation = scene['directions']['right']

  num_boxes = len(objs)

  boxes = np.zeros((1, num_boxes, 4))

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []

  for i, obj in enumerate(objs):
    [x, y, z] = obj['pixel_coords']

    [x1, y1, z1] = obj['3d_coords']

    cos_theta, sin_theta, _ = rotation

    x1 = x1 * cos_theta + y1* sin_theta
    y1 = x1 * -sin_theta + y1 * cos_theta


    height_d = 6.9 * z1 * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    if obj['shape'] == 'cylinder':
      d = 9.4 + y1
      h = 6.4
      s = z1

      height_u *= (s*(h/d + 1)) / ((s*(h/d + 1)) - (s*(h-s)/d))
      height_d = height_u * (h-s+d)/ (h + s + d)

      width_l *= 11/(10 + y1)
      width_r = width_l

    if obj['shape'] == 'cube':
      height_u *= 1.3 * 10 / (10 + y1)
      height_d = height_u
      width_l = height_u
      width_r = height_u
    
    obj_name = obj['size'] + ' ' + obj['color'] + ' ' + obj['material'] + ' ' + obj['shape']
    classes_text.append(obj_name.encode('utf8'))
    classes.append(names.index(obj_name) + 1)
    ymin.append((y - height_d)/320.0)
    ymax.append((y + height_u)/320.0)
    xmin.append((x - width_l)/480.0)
    xmax.append((x + width_r)/480.0)

  return xmin, ymin, xmax, ymax, classes, classes_text
