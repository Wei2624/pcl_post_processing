#include <iostream>
#include <typeinfo>
#include <Python.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

extern "C" int add_one(double* in, int len, int chl, double* out)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>),cloud_w (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> > M(in,len,chl);
  Eigen::MatrixXf Mf = M.cast <float> ();

  for(int k=0;k<Mf.rows();k++)
  {
    pcl::PointXYZ pt(Mf(k,0),Mf(k,1),Mf(k,2));
    cloud_w->push_back(pt);
  }
  std::cerr << cloud_w->points.size()<<std::endl;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;

  seg.setOptimizeCoefficients (true);

  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (0.01);

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  int i = 0, nr_points = (int) cloud_w->points.size ();
  float min_dis = 10000.0;
  Eigen::MatrixXf table_pcd;

  while (cloud_w->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_w);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the inliers
    extract.setInputCloud (cloud_w);
    extract.setIndices (inliers);
    extract.setNegative (false);//extract points only referenced by inliers
    extract.filter (*cloud_p);

    Eigen::MatrixXf pcd = cloud_p->getMatrixXfMap();
    float temp_min_dis = pcd.block(0,0,3,pcd.cols()).colwise().norm().rowwise().minCoeff()(0,0);
    if (temp_min_dis < min_dis)
    {
      table_pcd = pcd;
      min_dis = temp_min_dis;

    }
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_w.swap (cloud_f);
    i++;
  }
  for(i=0;i < table_pcd.cols();i++)
  {
    out[i] = (double)table_pcd(0,i);
    out[i+table_pcd.cols()] = (double)table_pcd(1,i);
    out[i+2*table_pcd.cols()] = (double)table_pcd(2,i);
  }

  return table_pcd.cols();
}

int main()
{
  return 0;
}




