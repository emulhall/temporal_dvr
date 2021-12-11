import numpy as np
import open3d as o3d

def visualize_3D(X, fname='test_point_cloud.ply'):
	"""
		Saves a 3D point cloud

		Parameters
		----------
		X - set of 3D points: numpy array of size (3,256,256)
		mask - set of valid 3D points: numpy array of size (256,256)
		fname - where to save the point cloud: String
	"""

	
	#Build point cloud
	pcd=o3d.geometry.PointCloud()
	pcd.points=o3d.utility.Vector3dVector(X)
	o3d.io.write_point_cloud(fname, pcd)

def visualize_3D_masked(X, mask, fname='test_point_cloud.ply'):
	"""
		Saves a 3D point cloud

		Parameters
		----------
		X - set of 3D points: numpy array of size (3,256,256)
		mask - set of valid 3D points: numpy array of size (256,256)
		fname - where to save the point cloud: String
	"""
	#Get valid points only for the point cloud
	valid_X_ind=np.argwhere(mask>0) # (N,2)
	valid_X=X[:,valid_X_ind[:,0], valid_X_ind[:,1]] # (3,N)
	valid_X=valid_X.T # (N,3)
	
	#Build point cloud
	pcd=o3d.geometry.PointCloud()
	pcd.points=o3d.utility.Vector3dVector(valid_X)
	o3d.io.write_point_cloud(fname, pcd)
