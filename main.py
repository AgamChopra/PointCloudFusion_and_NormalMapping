'''
Title: HW-4, CS-532 Spring 2022, main code file
Author: Agamdeep S. Chopra
Date: 04/28/2022
'''
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import utilities as util


def main(Out_path = ''):   
    '''
    #### PROBLEM 1 ####
    '''
    # Loading Images
    im_1 = np.asarray(Image.open(r'hw4_data\problem1\rgb1.png'))
    im_2 = np.asarray(Image.open(r'hw4_data\problem1\rgb2.png'))
    im_3 = np.asarray(Image.open(r'hw4_data\problem1\rgb3.png'))
    
    disp_1 = np.asarray(Image.open(r'hw4_data\problem1\depth1.png'))
    disp_2 = np.asarray(Image.open(r'hw4_data\problem1\depth2.png'))
    disp_3 = np.asarray(Image.open(r'hw4_data\problem1\depth3.png'))
    
    ## Part 1. Harris Corner Detection ##!!!
    im_1_100_list,_,im_1_ = util.top100corners(im_1,0.06)
    im_2_100_list,_,im_2_ = util.top100corners(im_2,0.06)
    im_3_100_list,_,im_3_ = util.top100corners(im_3,0.06)
    
    plt.imshow(im_1_)
    plt.show()
    plt.imshow(im_2_)
    plt.show()
    plt.imshow(im_3_)
    plt.show()
    
    pt_list_2d = np.asarray([im_1_100_list,im_2_100_list,im_3_100_list]) # (3,100,2)
    
    ## Part 2. Corners to 3D points ##!!!
    depth_list = np.asarray([disp_1,disp_2,disp_3]) #(3, 480, 640)
    S = 5000
    K = np.asarray([[525.0,0,319.5],[0,525.0,239.5],[0,0,1]])
    dxy = np.zeros((3,100))
    
    for i in range(100):
        dxy[0,i] = depth_list[0,pt_list_2d[0,i,0],pt_list_2d[0,i,1]]
        dxy[1,i] = depth_list[1,pt_list_2d[1,i,0],pt_list_2d[1,i,1]]
        dxy[2,i] = depth_list[2,pt_list_2d[2,i,0],pt_list_2d[2,i,1]]
        
    pt_list_3d = []
    pt_list_2d_clean = []
    
    for i in range(3):
        pt_list_3d.append([])
        pt_list_2d_clean.append([])
        for j in range(100):
            d = dxy[i,j]
            if d > 0:
                pt = (1/S) * d * (np.linalg.inv(K) @ np.array([pt_list_2d[i,j,0], pt_list_2d[i,j,1], 1.]))
                pt_list_3d[-1].append(pt)
                pt_list_2d_clean[-1].append(pt_list_2d[i,j])
                
    print(pt_list_3d)
    
    ## Part 3. Corner Matching ##!!!
    
    ranks = [util.rank_transform(im_1, 5),util.rank_transform(im_2, 5),util.rank_transform(im_3, 5)]
    
    plt.imshow(ranks[0])
    plt.show()
    plt.imshow(ranks[1])
    plt.show()
    plt.imshow(ranks[2])
    plt.show()
    
    dist_list_12 = util.SAD_dist(ranks[0], ranks[1], pt_list_2d_clean[0], pt_list_2d_clean[1])
    print(dist_list_12.shape) 
    
    dist_list_32 = util.SAD_dist(ranks[2], ranks[1], pt_list_2d_clean[2], pt_list_2d_clean[1]) 
    print(dist_list_32.shape) 
    
    min_idx_12, min_idx_32 = np.argmin(dist_list_12,axis=1), np.argmin(dist_list_32,axis=1)
    print(min_idx_12.shape, min_idx_32.shape)
    
    dist_ppx_12, dist_ppx_32 = [], []
    
    for i in range(len(min_idx_12)):
        dist_ppx_12.append(dist_list_12[i,min_idx_12[i]])
    dist_ppx_12 = np.asarray(dist_ppx_12)   
    
    for i in range(len(min_idx_32)): 
        dist_ppx_32.append(dist_list_32[i,min_idx_32[i]])
        
    dist_ppx_32 = np.asarray(dist_ppx_32)
    print(dist_ppx_12.shape,dist_ppx_32.shape) 
       
    mins_12_1 = np.argsort(dist_ppx_12)[:10]
    mins_32_3 = np.argsort(dist_ppx_32)[:10]
    mins_12_2 = min_idx_12[mins_12_1]
    mins_32_2 = min_idx_32[mins_32_3]
    
    top10_12, top10_32 = np.asarray([mins_12_1,mins_12_2]).T, np.asarray([mins_32_3,mins_32_2]).T # first col. = img1 or img3, 2nd col. = img2
    print(top10_12)
    print(top10_32)
    
    ## Part 4. Pose Estimation ##!!!   
    R12,T12 = util.RANSAC(pt_list_3d[0],pt_list_3d[1],top10_12,alpha = 0.0005, beta = 0.7) #Note: play around with alpha and beta to get best result. alpha = 0.005, beta = 0.8
    R32,T32 = util.RANSAC(pt_list_3d[2],pt_list_3d[1],top10_32,alpha = 0.005, beta = 0.8) #Note: play around with alpha and beta to get best result. alpha = 0.005, beta = 0.8
    
    ## Part 5.  Finis Coronat Opus ##!!!
    dxy = np.asarray([disp_1,disp_2,disp_3]) #(3, 480, 640)
    points = []
    color = []
    im_list = np.asarray([im_1,im_2,im_3])
    
    for i in range(dxy.shape[0]):
        for j in range(dxy.shape[1]):
            for k in range(dxy.shape[2]):
                d = dxy[i,j,k]
                if d > 0:
                    pt = (1/S) * d * (np.linalg.inv(K) @ np.array([j, k, 1.]))
                    if i == 0:
                        pt = R12 @ pt + T12
                    elif i == 2:
                        pt = R32 @ pt + T32
                    points.append(pt)
                    color.append(im_list[i,j,k]) 
                    
    point_cloud = np.asarray(points)
    color = np.asarray(color)
    util.save_point_cloud(point_cloud, color, Out_path, 'HW4_P1_point_cloud')
    
    '''
    #### PROBLEM 2 ####
    '''
    img = Image.open(r'hw4_data\problem2\rgbn.png')
    img = np.asarray(img) 
    
    plt.imshow(img)
    plt.show()
    
    depth = Image.open(r'hw4_data\problem2\depthn.png')
    depth = np.asarray(depth)
    
    plt.imshow(depth)
    plt.show()
    
    ### Part 1 ###
    points3d = []
    color = []
    mask = np.ones(depth.shape) * -1
    idx = 0
    
    for j in range(depth.shape[0]):
        for k in range(depth.shape[1]): 
            d = depth[j,k]
            if d > 0:
                pt = (1/S) * d * (np.linalg.inv(K) @ np.array([j, k, 1.]))
                points3d.append(pt)
                color.append(im_list[i,j,k])
                mask[j,k] = idx
                idx += 1
                
    plt.imshow(np.where(mask>-1,1,0),cmap = 'binary')
    plt.show()
    
    ### Part 2 ###
    normals = util.FastNormalApproximation(points3d,mask)
    
    ### Part 3 ###
    img_normal = util.normal2img(normals,mask)
    
    plt.imshow(img_normal)
    plt.show()
    
    ### Part 4 ###
    normals = util.FastNormalApproximationSimpleSmooth(points3d,mask)
    img_normal = util.normal2img(normals,mask)
    
    plt.imshow(img_normal)
    plt.show()
    
    
if __name__ == '__main__':
    main('p1_output')