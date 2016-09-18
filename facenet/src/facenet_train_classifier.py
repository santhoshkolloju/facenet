"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw

def main(args):
  
    network = importlib.import_module(args.model_def, 'inference')

    if args.model_name:
        subdir = args.model_name
        preload_model = True
    else:
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        preload_model = False
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    dataset = facenet.get_dataset(args.data_dir)
    train_set, test_set = facenet.split_dataset(dataset, 0.95, 'SPLIT_IMAGES')
    assert(len(train_set)==len(test_set))
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    
    # Read the file containing the pairs used for testing
    if args.lfw_dir:
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

    # Get the paths for the corresponding images
    if args.lfw_dir:
        paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        # Read data and apply label preserving distortions
        image_batch, label_batch, total_nrof_examples = facenet.read_and_augument_data(train_set, args.image_size, 
            args.batch_size, args.max_nrof_epochs, args.random_crop, args.random_flip)
        print('Total number of classes: %d' % len(train_set))
        print('Total number of examples: %d' % total_nrof_examples)
        
        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3), name='input')

        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learing_rate')

        # Build the inference graph
        logits, endpoints = network.inference(image_batch, len(train_set), args.keep_probability, 
            phase_train=phase_train_placeholder, weight_decay=args.weight_decay)
        
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.scalar_summary('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, label_batch, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.all_variables())

        embeddings = tf.nn.l2_normalize(endpoints['PreLogitsFlatten'], 1, 1e-10, name='embeddings')

        # Create a saver
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)
        tf.train.start_queue_runners(sess=sess)

        with sess.as_default():

            if preload_model:
                ckpt = tf.train.get_checkpoint_state(model_dir)
                #pylint: disable=maybe-no-member
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError('Checkpoint not found')

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                epoch = sess.run(global_step, feed_dict=None) // args.epoch_size
                # Train for one epoch
                step = train(args, sess, epoch, phase_train_placeholder, learning_rate_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses)
                
#                 # Visualize conv1 features
#                 w = sess.run('conv1_7x7/weights:0')
#                 nrof_rows = 8
#                 nrof_cols = 8
#                 features = np.zeros((1+(7+1)*nrof_rows,1+(7+1)*nrof_cols,3),dtype=np.float32)
#                 d = 7+1
#                 for i in range(nrof_rows):
#                     for j in range(nrof_cols):
#                         filt = w[:,:,:,i*nrof_cols+j]
#                         x_min = np.min(filt)
#                         x_max = np.max(filt)
#                         filt_norm =(filt - x_min) / (x_max - x_min)
#                         features[d*i+1:d*(i+1), d*j+1:d*(j+1), :] = filt_norm
#                 features_resize = misc.imresize(features, 8.0, 'nearest')
#                 misc.imsave(os.path.join(log_dir, 'features_epoch%d.png' % epoch), features_resize)

                if args.lfw_dir:
                    _, _, accuracy, val, val_std, far = lfw.validate(sess, 
                        paths, actual_issame, args.seed, args.batch_size,
                        images_placeholder, phase_train_placeholder, embeddings, nrof_folds=args.lfw_nrof_folds)
                    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                    # Add validation loss and accuracy to summary
                    summary = tf.Summary()
                    #pylint: disable=maybe-no-member
                    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
                    summary.value.add(tag='lfw/val_rate', simple_value=val)
                    summary_writer.add_summary(summary, step)

                # Save the model checkpoint
                print('Saving checkpoint')
                checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

#                 if (epoch % args.checkpoint_period == 0) or (epoch==args.max_nrof_epochs-1):
#                     precision = evaluate(network, test_set, model_dir, args.image_size, args.batch_size, args.moving_average_decay)
#                     summary = tf.Summary()
#                     #pylint: disable=maybe-no-member
#                     summary.value.add(tag='precision', simple_value=precision)
#                     summary_writer.add_summary(summary, step)
                
    return model_dir
  
def train(args, sess, epoch, phase_train_placeholder, learning_rate_placeholder, global_step, 
      loss, train_op, summary_op, summary_writer, regularization_losses):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file('../data/learning_rate_schedule.txt', epoch)
    while batch_number < args.epoch_size:
        # Perform training on the selected triplets
        train_time = 0
        i = 0
        while batch_number < args.epoch_size:
            start_time = time.time()
            feed_dict = {phase_train_placeholder: True, learning_rate_placeholder: lr}
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
            if (batch_number % 100 == 0):
                summary_str, step = sess.run([summary_op, global_step], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
            batch_number += 1
            i += 1
            train_time += duration
        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/total', simple_value=train_time)
        summary_writer.add_summary(summary, step)
    return step
  
def evaluate(network, test_set, model_dir, image_size,  batch_size, moving_average_decay):
  
    with tf.Graph().as_default():
  
        # Read data without augumentation
        image_batch, label_batch, total_nrof_examples = facenet.read_and_augument_data(test_set, image_size, 
            batch_size, 1, False, False)
        
        # Build the inference graph
        logits, _ = network.inference(image_batch, len(test_set), 1.0, 
            phase_train=False, weight_decay=0.0)
        
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, label_batch, 1)
    
        # Restore the moving average version of the learned variables
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
            sess.run(tf.initialize_local_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path: #pylint: disable=maybe-no-member
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)  #pylint: disable=maybe-no-member
            else:
                print('No checkpoint file found')
                return -1.0
        
            # Start the queue runners
            coord = tf.train.Coordinator()
            precision = -1.0
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
          
                true_count = 0
                nrof_samples = 0
                print('Running forward pass on test dataset')
                while nrof_samples+batch_size<total_nrof_examples and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    nrof_samples += np.size(predictions)
                    true_count += np.sum(predictions)
          
                # Compute precision @ 1.
                precision = true_count / nrof_samples
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
          
            except Exception as e:  # pylint: disable=broad-except
                print(e)
                coord.request_stop(e)
        
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
    
        return precision
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--model_name', type=str,
        help='Model directory name. Used when continuing training of an existing model. Leave empty to train new model.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/facescrub/fs_aligned:~/datasets/casia/casia-webface-aligned')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.nn4')
    parser.add_argument('--loss_type', type=str,
        help='Type of loss function to use', default='TRIPLETLOSS', choices=['TRIPLETLOSS', 'CLASSIFIER'])
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--checkpoint_period', type=int,
        help='The number of epochs between checkpoints', default=10)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--pool_type', type=str,
        help='The type of pooling to use for some of the inception layers', default='MAX', choices=['MAX', 'L2'])
    parser.add_argument('--use_lrn', 
        help='Enables Local Response Normalization after the first layers of the inception network.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))