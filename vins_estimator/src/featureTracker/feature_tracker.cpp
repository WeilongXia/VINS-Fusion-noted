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

#include "feature_tracker.h"

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

// At the position where the feature points have been tracked,
// set the corresponding position of the mask to 0.
// Corner detection is not repeated at this point, which can
// make the distribution of corners more uniform.
void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    // The feature points set is reordered according to the number
    // of tracking from large to small.
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
             return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// Track previous image and extract the feature points of the current frame.
/**
 * @brief   对图像使用光流法进行特征点跟踪
 * @Description createCLAHE() 对图像进行自适应直方图均衡化（暂时不用，注释）
 *              calcOpticalFlowPyrLK() LK金字塔光流法
 *              rejectWithF() 通过基本矩阵剔除outliers（暂时不用，注释）
 *              setMask() 对跟踪点进行排序，设置mask
 *              goodFeaturesToTrack() 添加特征点(shi-tomasi角点)，确保每帧都有足够的特征点
 *              undistortedPoints() 对角点图像坐标去畸变矫正，并计算每个角点的速度
 * @param[in]   _img, _img1 输入图像，左目和右目
 * @param[in]   _cur_time 当前时间（图像时间戳）
 * @return      map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame
 *              特征点ID，所观测到的相机ID，特征点在相机归一化平面坐标x,y,z，特征点在像素平面坐标u,v，特征点在归一化平面移动速度vx,vy
 */
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time,
                                                                                    const cv::Mat &_img,
                                                                                    const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;

    // histogram equalization to improve image contrast
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    // 此时cur_pts还保存的是上一帧图像中的特征点，所以把它清除
    cur_pts.clear();

    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // 1. Use the motion of the previous frame to estimate the position of the feature points
        // in the current frame. If the feature points has no speed, use the position of the points
        // in the previous frame directly.
        if (hasPrediction)
        {
            // 如果有预测特征点位置，将通过上一帧位置和速度预测的当前帧位置赋值给当前帧特征点
            cur_pts = predict_pts;
            // the layer of the pyramid is 1
            // 调用cv::calcOpticalFlowPyrLK()对前一帧的特征点prev_pts进行LK金字塔光流跟踪，得到cur_pts
            // status标记了从前一帧prev_img到cur_img特征点的跟踪状态，无法被追踪到的点标记为0
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);

            // succ_num is the number of tracked feature points
            int succ_num = 0;
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                    succ_num++;
            }
            // if the number of tracked feature points is less than 10, then
            // the number of pyramid layers is adjusted to 3
            if (succ_num < 10)
            {
                // 特征点成功跟踪数量少于10（说明相机运动剧烈），用3层金字塔重新跟踪
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            }
        }
        else
        {
            // 没预测特征点位置的话就用3层金字塔跟踪
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        }

        // reverse check
        // 2. LK optical flow tracks the feature points of the previous frame, forward and reverse
        if (FLOW_BACK)
        {
            // 反向跟踪，防止错误匹配
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts;
            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);
            // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21),
            // 3);
            for (size_t i = 0; i < status.size(); i++)
            {
                // the distance of matching pixel coordinates can not bigger than 0.5
                if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }

        for (int i = 0; i < int(cur_pts.size()); i++)
        {
            // check if the tracked feature points is in image boundary
            // 将位于图像边界外的点标记为0
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        }
        // 根据status,把跟踪失败的点剔除
        // 不仅要从当前帧数据cur_pts中剔除，还要从prev_pts中剔除
        // prev_pts和cur_pts中的特征点是一一对应的
        // 记录特征点ID的ids,和记录特征点跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        // printf("track cnt %d\n", (int)ids.size());
    }

    // 光流跟踪成功，特征点被成功跟踪的次数就加1
    // n数值表示被追踪的次数，数值越大，说明被追踪的就越久
    for (auto &n : track_cnt)
        n++;

    if (1)
    {
        // 3. For the matching feature points tracked by the LK optical flow in the front and current
        // frames, calculate the fundamental matrix, and use epipolar constraints to further eliminate
        // outliers.
        // 通过基本矩阵剔除outliers
        // rejectWithF();

        ROS_DEBUG("set mask begins");
        TicToc t_m;
        // 保证相邻的特征点之间要相隔30个pixel，设置mask，这样使特征点在整个图像上分布更加均匀
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        // 4. If there are not enough feature points, use corner points to make up the rest.
        // Update the number of feature points tracked.
        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        // 计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)
        {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            // extract shi-tomasi corner points by default
            /**
             *void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,
             角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝 //
             *   double  minDistance,
             返回角点之间欧式距离的最小值 //
             *   InputArray  mask = noArray(),
             和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点 //
             *   int blockSize = 3,
             计算协方差矩阵时的窗口大小 //
             *   bool useHarrisDetector = false,
             指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法 //
             *   double  k = 0.04 Harris角点检测需要的k值
             *)
             */
            cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        // 将新检测到的特征点n_pts添加到cur_pts中，id逐个加1，track_cnt初始化为1
        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        // printf("feature cnt after add %d\n", (int)ids.size());
    }

    // 5. Calculate the normalized camera plane coordinates of the feature points, and calculate the
    // moving speed relative to the previous frame.
    // 根据不同的相机模型去畸变校正和转换到归一化坐标系上，并计算速度
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // If it is stereo, perform left and right matching
    if (!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if (!cur_pts.empty())
        {
            // printf("stereo image; track feature on right image\n");
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            // cur left ---- cur right
            cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
            if (FLOW_BACK)
            {
                cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err,
                                         cv::Size(21, 21), 3);
                for (size_t i = 0; i < status.size(); i++)
                {
                    if (status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) &&
                        distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    // 7. Display, the feature points in the left image are tracked by color(less red, more blue), draw
    // an arrow pointing to the position of the feature points in the previous frame, if it is stereo,
    // draw green points in the right image.
    // 如果需要可视化，将特征点绘制在当前帧左右目图像上
    if (SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    // 将当前帧数据赋值给上一帧
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    // 可视化时会画出上一帧特征点到当前帧特征点的对应连线，所以需要记录上一帧特征点的像素坐标并传入drawTrack()函数中
    prevLeftPtsMap.clear();
    for (size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    // 6. Save the feature points data of the current frame(normalized camera plane coordinates, pixel
    // coordinates, normalized camera plane moving speed).
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y, z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        // observed by the left eye if camera_id = 0
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y, z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            // observed by the right eye if camera_id = 1
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
    }

    // printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            // 首先对特征点坐标进行归一化处理，并进行畸变校正
            // 然后再将归一化坐标转换成像素坐标
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        // 调用OpenCV函数进行RANSAC，然后再剔除外点
        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 &&
            pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) =
                cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0,
            // 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        // pixel coordinates of the feature points
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        // use the cooresponding camera model to calculate coordinates in camera frame
        // (including undistortion)
        cam->liftProjective(a, b);
        // obtain normalized camera plane coordinates
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

// Calculate the moving speed of the feature point in current normalized camera plane
// in x and y directions
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                                map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;

        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    // int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            // cv::Point2f leftPt = curLeftPtsTrackRight[i];
            // cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    // draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    // printf("predict pts size %d \n", (int)predict_pts_debug.size());

    // cv::Mat imCur2Compress;
    // cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        // printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}

void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if (itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}