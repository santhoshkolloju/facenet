
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import pickle
from flask import Flask,request
from flask import jsonify


app = Flask(__name__)


sess = tf.Session()
facenet.load_model("../20180408-102900.pb")
graph = tf.get_default_graph()
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

def main(args,image_files):

    images,images_names = load_and_align_data(image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    

    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)
            
    nrof_images = len(images)
    base_image_index = images_names.index("base_image")
    
    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i, images_names[i]))
    print('')
    
    res={}
    
    for i in range(nrof_images):
        if i!= base_image_index:
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[base_image_index,:], emb[i,:]))))
            if dist > float(args.threshold):
                out = {"distance": str(dist) ,'SAME' : "No"}
            else:
                out = {"distance": str(dist) ,'SAME' : "Yes"}
            
            res[images_names[i]] = out
    return res
            
            
            
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    img_list = []
    image_names=[]
    for image_name,image_data in image_paths.items():
        img = misc.imread(image_data, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          #image_paths.remove(image)
          print("can't detect face, remove ", image_name)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        image_names.append(image_name)
    images = np.stack(img_list)
    return images,image_names




@app.route("/scoreimages",methods=['GET','POST'])
def score_images():
    if request.method=='POST':
        images_to_be_compared=[]
        if len(request.files)>1:
            if 'base_image' in request.files:
                base_image=request.files['base_image']
                #load_and_align_data(request.files)
                res = main(parse_arguments(sys.argv[1:]),request.files)
                return jsonify(res)
            else:
                print("INPUT IMAGE NOT FOUND")
                return -1
            
            
            
    


@app.route("/hello",methods=['GET','POST'])
def hello():
    return "helloworld"


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    
    parser.add_argument('--threshold', type=float,
        help='Threshold to classify as SAME IMAGE OR DIFFERENT IMAGE default is 0.85', default=0.85)
    
    #parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 1120,debug = True)
    
