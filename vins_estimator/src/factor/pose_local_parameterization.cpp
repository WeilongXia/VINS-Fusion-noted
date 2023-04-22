/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    // 位置直接线性相加，姿态右乘更新量
    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}

// 位姿向量的维度是7，但是实际上位姿的自由度是6，在优化过程中引入局部参数化来消除冗余自由度
// ComputeJacobian()计算了位姿向量关于一个6维增量的Jacobi矩阵
// 我们关注的是优化过程中的增量(delta)对位姿向量的影响
// 左上角的6*6子矩阵是单位阵，表示位置分量的线性变化
// 右下角的1*6子矩阵是0矩阵，表示四元数标量分量(w分量)在优化过程中保持不变
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
