import pickle
import numpy as np
import sys
import argparse


with open("../Face_distance_matrix/face_distance.pkl",'rb') as f:
    dis = pickle.load(f)

    
with open("../Face_distance_matrix/image_paths.pkl",'rb') as f:
    paths = pickle.load(f)
    
    
similar_images = {}


def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold

def main(args):
    similar_images_thresholds=[]
    acc=args.accuracy
    acc_d=args.dis_accuracy
    for i,p in enumerate(paths):
        folder = p.split("/")[-2]
        if folder not in similar_images:
            similar_images[folder] =[]
            similar_images[folder].append(i)
        else:
            similar_images[folder].append(i)
    for folder,indices in similar_images.items():
        tri_ind = np.triu_indices(len(indices),1)
        sim = np.triu(dis[np.ix_(indices,indices)])
        dis[np.ix_(indices,indices)] = -100
        print(folder)
        similar_images_thresholds.extend(sim[tri_ind].flatten())
    print("Similar Image Thresholds",similar_images_thresholds)
    threshold=find_threshold(similar_images_thresholds,acc)
    dis_sim = [s for s in dis.flatten() if s!=-100 ]
    threshold_dis = find_threshold(dis_sim,acc_d)
    return threshold,threshold_dis


        
        


    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('accuracy', type=int, 
        help='Find the threshold where the given accuracy of the model should be')
    parser.add_argument('dis_accuracy', type=int, 
        help='Find the threshold where the given accuracy of the model should be for rejecting the dis similar images')
    
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    threshold,threshold_dis = main(parse_arguments(sys.argv[1:]))    
    print("Threshold for given accuracy is ",threshold)
    print("Threshold for given accuracy for finding the dissimialr images ",threshold_dis)
    
    
    
