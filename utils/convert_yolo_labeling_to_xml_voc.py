"""
Convert inputed yolo format labeling into xml voc format
"""



import os
import glob
import xml.etree.ElementTree as ET
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2 as cv


flags.DEFINE_string('images', './data/images', 'path to input images')
flags.DEFINE_string('output', './data/xm_annotation', 'path to xml output folder')
flags.DEFINE_string('annot_path', './data/yolo_annot', 'path to yolo annotation')
flags.DEFINE_string('classes', './data/class_names.txt', 'file path of class names')



def get_bbox_from_yolo_labels(class_id, image_width, image_height, bbox):
  """
  x_center, y_center, yolo_width, yolo_height = bbox[0], bbox[1], bbox[2], bbox[3]  
  >> inp bbox, width, heigh x_center, y_center, yolo_height, yolo_width
  >> return voc bbox (xmin, ymin, xmax, ymax)
  """
  w_half_len = (bbox[2] * image_width)/2
  h_half_len = (bbox[3] * image_height)/2
  xmin = int(bbox[0] * image_width - w_half_len)
  ymin = int(bbox[1] * image_height - h_half_len)
  xmax = int(bbox[0] * image_width + w_half_len)
  ymax = int(bbox[1] * image_height + h_half_len)
  class_id = int(class_id)
  return (class_id,xmin, ymin, xmax, ymax)

def xml_create_from_yolo():

    """
    >> input: image and yolo annotation path, output path, class names
    >> output: xml voc format annotations
    """
    image_path = FLAGS.images  
    img_path=os.listdir(image_path)
    
    ids=[]
    for x in img_path:
      if not os.path.isdir(x):
        ids.append(x.split('.')[0])

    yolo_txt_path = os.path.join(FLAGS.annot_path, '%s.txt')
    imgpath = os.path.join(FLAGS.images,  '%s.jpg')    
    outpath = os.path.join(FLAGS.output, '%s.xml')
    class_names_file = FLAGS.classes
    class_names=[]
    with open(class_names_file) as lab:
      for l in lab:
       class_names.append(l.rstrip('\n'))
    
   

    for i in range(len(ids)):
        img_id = ids[i] 
        if img_id == "classes":
            continue
        if os.path.exists(outpath % img_id):
            continue
        print(imgpath % img_id)
        img= cv.imread(imgpath % img_id)
        height, width, channels = img.shape

        node_root = ET.Element('annotation')
        node_folder = ET.SubElement(node_root, 'folder')
        node_folder.text = 'train'
        img_name = img_id + '.jpg'
    
        node_filename = ET.SubElement(node_root, 'filename')
        node_filename.text = img_name
        
        node_source= ET.SubElement(node_root, 'source')
        node_database = ET.SubElement(node_source, 'database')
        node_database.text = 'Armplate'
        
        node_size = ET.SubElement(node_root, 'size')
        node_width = ET.SubElement(node_size, 'width')
        node_width.text = str(width)
    
        node_height = ET.SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = ET.SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = ET.SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = (yolo_txt_path % img_id)

        if os.path.exists(target):
            label_norm= np.loadtxt(target).reshape(-1, 5)                    
            for i in range(len(label_norm)):
                labels_conv = label_norm[i]                
                bbox = labels_conv[1:]                
                new_label = get_bbox_from_yolo_labels(labels_conv[0], width, height, bbox)
                node_object = ET.SubElement(node_root, 'object')
                node_name = ET.SubElement(node_object, 'name')                
                node_name.text = class_names[new_label[0]]
                
                node_pose = ET.SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'                
                
                node_truncated = ET.SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = ET.SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = ET.SubElement(node_object, 'bndbox')
                node_xmin = ET.SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(new_label[1])
                node_ymin = ET.SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(new_label[2])
                node_xmax = ET.SubElement(node_bndbox, 'xmax')
                node_xmax.text =  str(new_label[3])
                node_ymax = ET.SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(new_label[4])
                xml = ET.tostring(node_root, encoding="unicode")  
                
        
        f =  open(outpath % img_id, "w")  
        f.write(xml)
        f.close()

        
def main(_argv):
  xml_create_from_yolo()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
