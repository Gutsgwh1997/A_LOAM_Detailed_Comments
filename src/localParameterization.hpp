#include <ceres/ceres.h>

namespace ceres{

class EigenQuaternionParameterization : public ceres::LocalParameterization {
   public:
    virtual ~EigenQuaternionParameterization() {}
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
    bool ComputeJacobian(const double* x, double* jacobian) const override;
    int GlobalSize() const override { return 4; }
    int LocalSize() const override { return 3; }
};


// 定义广义的加法
bool EigenQuaternionParameterization::Plus(const double* x_ptr, const double* delta, double* x_plus_delta_ptr) const {
    // 四元数在内存中的顺序是x,y,z,w
    Eigen::Map<Eigen::Quaterniond> x_plus_delta(x_plus_delta_ptr);
    Eigen::Map<const Eigen::Quaterniond> x(x_ptr);

    const double norm_delta = sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    if (norm_delta > 0.0) {
        const double sin_delta_by_delta = sin(norm_delta) / norm_delta;
        // Note, in the constructor w is first.
        Eigen::Quaterniond delta_q(cos(norm_delta), sin_delta_by_delta * delta[0], sin_delta_by_delta * delta[1], sin_delta_by_delta * delta[2]);
        x_plus_delta = delta_q * x;
    } else {
        x_plus_delta = x;
    }

    return true;
}

// Jacobian的大小是4*3的
bool EigenQuaternionParameterization::ComputeJacobian(const double* x, double* jacobian) const {
    jacobian[0] =  x[3];  jacobian[1]  =  x[2];  jacobian[2]  = -x[1];
    jacobian[3] = -x[2];  jacobian[4]  =  x[3];  jacobian[5]  =  x[0];
    jacobian[6] =  x[1];  jacobian[7]  = -x[0];  jacobian[8]  =  x[3];
    jacobian[9] = -x[0];  jacobian[10] = -x[1];  jacobian[11] = -x[2];
    
    return true;
}
}
