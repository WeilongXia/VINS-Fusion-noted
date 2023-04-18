/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <ros/ros.h>
#include <stdio.h>
#include <thread>
// #include <time.h>

// 全局变量，生命周期直至整个程序结束
Estimator estimator;

// imu_buf and feature_buf are not used
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;

queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

// int cnt_pose = 0;
// int cnt_pic = 0;
// ros::Subscriber dense_sub;
// ros::Subscriber rbg_sub;
// ros::Subscriber pose_sub;
// ofstream pose_txt_file;
// std::mutex save_buf;
// cv::Mat dense_img;
// double x, y, z, qx, qy, qz, qw;

bool first_receive_gt = true;
Eigen::Matrix4d gt_to_estimate;

// void rgb_callback(const sensor_msgs::ImageConstPtr &rgb_msg)
// {
//     save_buf.lock();
//     cnt_pic++;
//     if (cnt_pic % 180 == 0)
//     {
//         cv_bridge::CvImagePtr cv_ptr;
//         cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
//         cv::imwrite("/home/xwl/catkin_ws/src/VINS-Fusion-noted/data/color/" + to_string(cnt_pic / 180) + ".png",
//                     cv_ptr->image);
//         cv::imwrite("/home/xwl/catkin_ws/src/VINS-Fusion-noted/data/depth/" + to_string(cnt_pic / 180) + ".png",
//                     dense_img);
//         pose_txt_file << x << setprecision(4) << "  " << y << setprecision(4) << "  " << z << setprecision(4) << "  "
//                       << qx << setprecision(4) << "  " << qy << setprecision(4) << "  " << qz << setprecision(4) << "
//                       "
//                       << qw << std::endl;
//     }
//     save_buf.unlock();
// }

// void dense_callback(const sensor_msgs::ImageConstPtr &dense_msg)
// {
//     // if (cnt_pic % 180 == 0)
//     // {
//     cv_bridge::CvImagePtr cv_ptr;
//     cv_ptr = cv_bridge::toCvCopy(dense_msg, sensor_msgs::image_encodings::TYPE_16UC1);
//     dense_img = cv_ptr->image;
//     // cv::imwrite("/home/xwl/catkin_ws/src/VINS-Fusion-noted/data/depth/" + to_string(cnt_pic / 180) + ".png",
//     // cv_ptr->image);
//     // }
// }

// void pose_callback(const nav_msgs::OdometryConstPtr &pose_msg)
// {
//     // cnt_pose++;
//     // if (cnt_pic % 90 == 0)
//     // {
//     x = pose_msg->pose.pose.position.x;
//     y = pose_msg->pose.pose.position.y;
//     z = pose_msg->pose.pose.position.z;
//     qx = pose_msg->pose.pose.orientation.x;
//     qy = pose_msg->pose.pose.orientation.y;
//     qz = pose_msg->pose.pose.orientation.z;
//     qw = pose_msg->pose.pose.orientation.w;
//     // pose_txt_file << pose_msg->pose.pose.position.x << setprecision(4) << "  " << pose_msg->pose.pose.position.y
//     //               << setprecision(4) << "  " << pose_msg->pose.pose.position.z << setprecision(4) << "  "
//     //               << pose_msg->pose.pose.orientation.x << setprecision(4) << "  " <<
//     //               pose_msg->pose.pose.orientation.y
//     //               << setprecision(4) << "  " << pose_msg->pose.pose.orientation.z << setprecision(4) << "  "
//     //               << pose_msg->pose.pose.orientation.w << std::endl;
//     // }
// }

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
// the duration threshold is 0.003s
void sync_process()
{
    while (1)
    {
        if (STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // 0.003s sync tolerance
                // the older the frame is, the smaller it's timestamp will be
                // queue: first in first out
                if (time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if (time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    // printf("find img0 and img1\n");
                }
            }
            m_buf.unlock();
            if (!image0.empty())
                estimator.inputImage(time, image0, image1);
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            m_buf.unlock();
            if (!image.empty())
                estimator.inputImage(time, image);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    // double dx = -imu_msg->linear_acceleration.z;
    // double dy = imu_msg->linear_acceleration.y;
    // double dz = imu_msg->linear_acceleration.x;
    // double rx = -imu_msg->angular_velocity.z;
    // double ry = imu_msg->angular_velocity.y;
    // double rz = imu_msg->angular_velocity.x;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    // for each feature points, it can be observed by several cameras.
    // map.key is the feature id, map.value is the camera id and the
    // pixel coordinate in the correspond camera.
    // feature id, camera id, normalized coordinate-pixel coordinate-xy velocity
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if (feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            // printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        // x, y, z is in the normalized plane of the camera.
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        // ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        // ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        // ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        // ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

void mocap_callback(const geometry_msgs::PoseStamped::ConstPtr &mocap_msg_ptr)
{
    ofstream foutC(MOCAP_RESULT_PATH, ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(9);
    foutC << mocap_msg_ptr->header.stamp.toSec() << ",";
    foutC.precision(5);
    foutC << mocap_msg_ptr->pose.position.x << "," << mocap_msg_ptr->pose.position.y << ","
          << mocap_msg_ptr->pose.position.z << "," << mocap_msg_ptr->pose.orientation.x << ","
          << mocap_msg_ptr->pose.orientation.y << "," << mocap_msg_ptr->pose.orientation.z << ","
          << mocap_msg_ptr->pose.orientation.w << endl;
    foutC.close();

    if (first_receive_gt)
    {
        Eigen::Matrix4d estimate_to_gt;
        estimate_to_gt.setIdentity(4, 4);
        Eigen::Quaterniond q(mocap_msg_ptr->pose.orientation.w, mocap_msg_ptr->pose.orientation.x,
                             mocap_msg_ptr->pose.orientation.y, mocap_msg_ptr->pose.orientation.z);
        estimate_to_gt.block<3, 3>(0, 0) = q.toRotationMatrix();
        estimate_to_gt.block<3, 1>(0, 3) << mocap_msg_ptr->pose.position.x, mocap_msg_ptr->pose.position.y,
            mocap_msg_ptr->pose.position.z;
        gt_to_estimate = estimate_to_gt.inverse();

        Eigen::Matrix4d gt_transformed = gt_to_estimate * estimate_to_gt;

        std::cout << "estimate_to_gt: " << estimate_to_gt << std::endl;
        std::cout << "gt_to_estimate: " << gt_to_estimate << std::endl;
        std::cout << "gt_transformed: " << gt_transformed << std::endl;

        Eigen::Quaterniond q_gt_estimate(gt_transformed.block<3, 3>(0, 0));

        nav_msgs::Odometry gt_pub;
        gt_pub.pose.pose.position.x = gt_transformed(0, 3);
        gt_pub.pose.pose.position.y = gt_transformed(1, 3);
        gt_pub.pose.pose.position.z = gt_transformed(2, 3);
        gt_pub.pose.pose.orientation.w = q_gt_estimate.w();
        gt_pub.pose.pose.orientation.x = q_gt_estimate.x();
        gt_pub.pose.pose.orientation.y = q_gt_estimate.y();
        gt_pub.pose.pose.orientation.z = q_gt_estimate.z();

        // publish mocap_msg_ptr aligned
        pubGt(gt_pub);

        first_receive_gt = false;
    }
    else
    {
        Eigen::Matrix4d estimate_to_gt;
        estimate_to_gt.setIdentity(4, 4);
        Eigen::Quaterniond q(mocap_msg_ptr->pose.orientation.w, mocap_msg_ptr->pose.orientation.x,
                             mocap_msg_ptr->pose.orientation.y, mocap_msg_ptr->pose.orientation.z);
        estimate_to_gt.block<3, 3>(0, 0) = q.toRotationMatrix();
        estimate_to_gt.block<3, 1>(0, 3) << mocap_msg_ptr->pose.position.x, mocap_msg_ptr->pose.position.y,
            mocap_msg_ptr->pose.position.z;

        Eigen::Matrix4d gt_transformed = gt_to_estimate * estimate_to_gt;

        Eigen::Quaterniond q_gt_estimate(gt_transformed.block<3, 3>(0, 0));

        nav_msgs::Odometry gt_pub;
        gt_pub.pose.pose.position.x = gt_transformed(0, 3);
        gt_pub.pose.pose.position.y = gt_transformed(1, 3);
        gt_pub.pose.pose.position.z = gt_transformed(2, 3);
        gt_pub.pose.pose.orientation.w = q_gt_estimate.w();
        gt_pub.pose.pose.orientation.x = q_gt_estimate.x();
        gt_pub.pose.pose.orientation.y = q_gt_estimate.y();
        gt_pub.pose.pose.orientation.z = q_gt_estimate.z();

        // publish mocap_msg_ptr aligned
        pubGt(gt_pub);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu;
    if (USE_IMU)
    {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1;
    if (STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

    // rbg_sub = n.subscribe("/camera/color/image_raw", 100, rgb_callback);
    // dense_sub = n.subscribe("/camera/depth/image_rect_raw", 100, dense_callback);
    // pose_sub = n.subscribe("/vins_estimator/odometry", 100, pose_callback);
    // pose_txt_file.open("/home/xwl/catkin_ws/src/VINS-Fusion-noted/data/pose.txt", ios::out | ios::trunc);
    // pose_txt_file << fixed;

    // subscribe groundtruth，and align it with estimated pose.
    // ros::Subscriber sub_ground_truth = n.subscribe("/drone1/ground_truth/ground_truth_odom", 100, gt_pub_callback);

    ros::Subscriber sub_mocap = n.subscribe(MOCAP_TOPIC, 100, mocap_callback);

    std::thread sync_thread{sync_process};
    ros::spin();

    // pose_txt_file.close();
    // std::cout << "finish" << std::endl;

    return 0;
}
