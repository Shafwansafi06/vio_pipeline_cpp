#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Dense>

// ============================================================================
// 1. OFFICIAL OPENVINS-STYLE STRUCTURES (Polymorphic, std::shared_ptr, Heap)
// ============================================================================
namespace openvins_style {

class Type {
public:
    virtual ~Type() = default;
    virtual int size() const = 0;
    virtual Eigen::VectorXd value() const = 0;
    virtual void set_value(const Eigen::VectorXd& val) = 0;
};

class IMU : public Type {
public:
    Eigen::Matrix<double, 16, 1> val;
    IMU() {
        val.setZero();
        val(3) = 1.0; // w = 1 for identity quat
    }
    int size() const override { return 15; }
    Eigen::VectorXd value() const override { return val; }
    void set_value(const Eigen::VectorXd& v) override { val = v; }
};

class PoseJPL : public Type {
public:
    Eigen::Matrix<double, 7, 1> val;
    PoseJPL() {
        val.setZero();
        val(3) = 1.0; // identity quat
    }
    int size() const override { return 6; }
    Eigen::VectorXd value() const override { return val; }
    void set_value(const Eigen::VectorXd& v) override { val = v; }
};

struct State {
    std::shared_ptr<IMU> imu;
    std::vector<std::shared_ptr<Type>> variables;
    Eigen::MatrixXd Cov;
    double timestamp;

    State() : timestamp(0.0) {
        imu = std::make_shared<IMU>();
        variables.push_back(imu);
        Cov = Eigen::MatrixXd::Zero(15, 15);
    }
};

inline Eigen::Matrix4d Omega(const Eigen::Vector3d& w) {
    Eigen::Matrix4d mat;
    mat <<    0,  w(2), -w(1),  w(0),
          -w(2),     0,  w(0),  w(1),
           w(1), -w(0),     0,  w(2),
          -w(0), -w(1), -w(2),     0;
    return mat;
}

inline Eigen::Matrix3d quat_2_Rot(const Eigen::Vector4d& q) {
    double q1 = q(0);
    double q2 = q(1);
    double q3 = q(2);
    double q4 = q(3);
    Eigen::Matrix3d R;
    R << q4*q4 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4),
         2*(q1*q2 - q3*q4), q4*q4 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 + q1*q4),
         2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), q4*q4 - q1*q1 - q2*q2 + q3*q3;
    return R;
}

inline Eigen::Vector4d quat_multiply(const Eigen::Vector4d& q, const Eigen::Vector4d& p) {
    Eigen::Vector4d r;
    r(0) = q(3)*p(0) + q(2)*p(1) - q(1)*p(2) + q(0)*p(3);
    r(1) = -q(2)*p(0) + q(3)*p(1) + q(0)*p(2) + q(1)*p(3);
    r(2) = q(1)*p(0) - q(0)*p(1) + q(3)*p(2) + q(2)*p(3);
    r(3) = -q(0)*p(0) - q(1)*p(1) - q(2)*p(2) + q(3)*p(3);
    return r;
}

// Predict mean using RK4 (Open_VINS style)
void predict_mean_rk4(const State& state, double dt,
                      const Eigen::Vector3d& w_hat1, const Eigen::Vector3d& a_hat1,
                      const Eigen::Vector3d& w_hat2, const Eigen::Vector3d& a_hat2,
                      Eigen::Vector4d& new_q, Eigen::Vector3d& new_v, Eigen::Vector3d& new_p) {
    Eigen::Vector3d w_alpha = (w_hat2 - w_hat1) / dt;
    Eigen::Vector3d a_jerk = (a_hat2 - a_hat1) / dt;

    Eigen::VectorXd imu_val = state.imu->value();
    Eigen::Vector4d q_0 = imu_val.head<4>();
    Eigen::Vector3d p_0 = imu_val.segment<3>(4);
    Eigen::Vector3d v_0 = imu_val.segment<3>(7);

    Eigen::Vector4d dq_0(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector3d gravity(0.0, 0.0, 9.81);

    auto compute_deriv = [&](const Eigen::Vector4d& q, const Eigen::Vector3d& v, const Eigen::Vector3d& w, const Eigen::Vector3d& a,
                             Eigen::Vector4d& q_dot, Eigen::Vector3d& p_dot, Eigen::Vector3d& v_dot) {
        q_dot = 0.5 * Omega(w) * q;
        p_dot = v;
        Eigen::Matrix3d R = quat_2_Rot(quat_multiply(q, q_0));
        v_dot = R.transpose() * a - gravity;
    };

    Eigen::Vector4d k1_q_dot; Eigen::Vector3d k1_p_dot, k1_v_dot;
    compute_deriv(dq_0, v_0, w_hat1, a_hat1, k1_q_dot, k1_p_dot, k1_v_dot);
    Eigen::Vector4d k1_q = k1_q_dot * dt; Eigen::Vector3d k1_p = k1_p_dot * dt, k1_v = k1_v_dot * dt;

    Eigen::Vector3d w_mid = w_hat1 + 0.5 * w_alpha * dt;
    Eigen::Vector3d a_mid = a_hat1 + 0.5 * a_jerk * dt;

    Eigen::Vector4d dq_1 = (dq_0 + 0.5 * k1_q).normalized();
    Eigen::Vector3d v_1 = v_0 + 0.5 * k1_v;
    Eigen::Vector4d k2_q_dot; Eigen::Vector3d k2_p_dot, k2_v_dot;
    compute_deriv(dq_1, v_1, w_mid, a_mid, k2_q_dot, k2_p_dot, k2_v_dot);
    Eigen::Vector4d k2_q = k2_q_dot * dt; Eigen::Vector3d k2_p = k2_p_dot * dt, k2_v = k2_v_dot * dt;

    Eigen::Vector4d dq_2 = (dq_0 + 0.5 * k2_q).normalized();
    Eigen::Vector3d v_2 = v_0 + 0.5 * k2_v;
    Eigen::Vector4d k3_q_dot; Eigen::Vector3d k3_p_dot, k3_v_dot;
    compute_deriv(dq_2, v_2, w_mid, a_mid, k3_q_dot, k3_p_dot, k3_v_dot);
    Eigen::Vector4d k3_q = k3_q_dot * dt; Eigen::Vector3d k3_p = k3_p_dot * dt, k3_v = k3_v_dot * dt;

    Eigen::Vector4d dq_3 = (dq_0 + k3_q).normalized();
    Eigen::Vector3d v_3 = v_0 + k3_v;
    Eigen::Vector4d k4_q_dot; Eigen::Vector3d k4_p_dot, k4_v_dot;
    compute_deriv(dq_3, v_3, w_hat2, a_hat2, k4_q_dot, k4_p_dot, k4_v_dot);
    Eigen::Vector4d k4_q = k4_q_dot * dt; Eigen::Vector3d k4_p = k4_p_dot * dt, k4_v = k4_v_dot * dt;

    Eigen::Vector4d dq = (dq_0 + (1.0 / 6.0) * k1_q + (1.0 / 3.0) * k2_q + (1.0 / 3.0) * k3_q + (1.0 / 6.0) * k4_q).normalized();
    new_q = quat_multiply(dq, q_0);
    new_p = p_0 + (1.0 / 6.0) * k1_p + (1.0 / 3.0) * k2_p + (1.0 / 3.0) * k3_p + (1.0 / 6.0) * k4_p;
    new_v = v_0 + (1.0 / 6.0) * k1_v + (1.0 / 3.0) * k2_v + (1.0 / 3.0) * k3_v + (1.0 / 6.0) * k4_v;
}

// Perform clone augmentation (requires heap allocation + dynamic matrix resize)
void augment_clone(State& state) {
    state.variables.push_back(std::make_shared<PoseJPL>());
    int prev_size = state.Cov.rows();
    state.Cov.conservativeResize(prev_size + 6, prev_size + 6);
    state.Cov.block(prev_size, 0, 6, prev_size).setZero();
    state.Cov.block(0, prev_size, prev_size, 6).setZero();
    state.Cov.block(prev_size, prev_size, 6, 6).setIdentity();

    // Marginalize oldest clone if sliding window is full
    if (state.variables.size() > 12) {
        state.variables.erase(state.variables.begin() + 1);
        int new_size = prev_size; // keeps size constant
        state.Cov.conservativeResize(new_size, new_size);
    }
}

// EKF Update (Open_VINS style with dynamic allocations)
void update_msckf(State& state) {
    int state_sz = state.Cov.rows();
    
    // Dynamic allocations of Jacobians
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(22, state_sz);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(22, 22) * 1e-3;
    
    Eigen::MatrixXd S = H * state.Cov * H.transpose() + R;
    Eigen::MatrixXd K = state.Cov * H.transpose() * S.inverse();
    state.Cov -= K * H * state.Cov;
}

} // namespace openvins_style

// ============================================================================
// 2. OUR PORTED C++17 DOD STATE STRUCTURES (Flat, Contiguous, Zero Allocation)
// ============================================================================
namespace our_style {

enum class VariableType { IMU, PoseJPL };

struct Variable {
    VariableType type;
    int id;
    int size;
    int storage_size;
    double value[16];
    double value_fej[16];
};

// Max state size: 15 (IMU) + 12*6 (clones) = 87
constexpr int MAX_STATE = 87;

struct State {
    Variable imu;
    Variable clones[12];
    int num_clones;
    int num_variables;
    Variable* variables[100];
    Eigen::Matrix<double, MAX_STATE, MAX_STATE> Cov;
    double timestamp;

    State() : num_clones(0), num_variables(0), timestamp(0.0) {
        imu.type = VariableType::IMU;
        imu.id = 0;
        imu.size = 15;
        imu.storage_size = 16;
        for (int i = 0; i < 16; ++i) {
            imu.value[i] = 0.0;
            imu.value_fej[i] = 0.0;
        }
        imu.value[3] = 1.0; // identity quat
        variables[num_variables++] = &imu;
        Cov.setZero();
    }
};

inline Eigen::Matrix4d Omega(const Eigen::Vector3d& w) {
    Eigen::Matrix4d mat;
    mat <<    0,  w(2), -w(1),  w(0),
          -w(2),     0,  w(0),  w(1),
           w(1), -w(0),     0,  w(2),
          -w(0), -w(1), -w(2),     0;
    return mat;
}

inline Eigen::Matrix3d quat_2_Rot(const Eigen::Vector4d& q) {
    double q1 = q(0);
    double q2 = q(1);
    double q3 = q(2);
    double q4 = q(3);
    Eigen::Matrix3d R;
    R << q4*q4 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4),
         2*(q1*q2 - q3*q4), q4*q4 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 + q1*q4),
         2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), q4*q4 - q1*q1 - q2*q2 + q3*q3;
    return R;
}

inline Eigen::Vector4d quat_multiply(const Eigen::Vector4d& q, const Eigen::Vector4d& p) {
    Eigen::Vector4d r;
    r(0) = q(3)*p(0) + q(2)*p(1) - q(1)*p(2) + q(0)*p(3);
    r(1) = -q(2)*p(0) + q(3)*p(1) + q(0)*p(2) + q(1)*p(3);
    r(2) = q(1)*p(0) - q(0)*p(1) + q(3)*p(2) + q(2)*p(3);
    r(3) = -q(0)*p(0) - q(1)*p(1) - q(2)*p(2) + q(3)*p(3);
    return r;
}

// Predict mean using RK4 (Our DOD style)
void predict_mean_rk4(const State& state, double dt,
                      const Eigen::Vector3d& w_hat1, const Eigen::Vector3d& a_hat1,
                      const Eigen::Vector3d& w_hat2, const Eigen::Vector3d& a_hat2,
                      Eigen::Vector4d& new_q, Eigen::Vector3d& new_v, Eigen::Vector3d& new_p) {
    Eigen::Vector3d w_alpha = (w_hat2 - w_hat1) / dt;
    Eigen::Vector3d a_jerk = (a_hat2 - a_hat1) / dt;

    Eigen::Vector4d q_0(state.imu.value[0], state.imu.value[1], state.imu.value[2], state.imu.value[3]);
    Eigen::Vector3d p_0(state.imu.value[4], state.imu.value[5], state.imu.value[6]);
    Eigen::Vector3d v_0(state.imu.value[7], state.imu.value[8], state.imu.value[9]);

    Eigen::Vector4d dq_0(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector3d gravity(0.0, 0.0, 9.81);

    auto compute_deriv = [&](const Eigen::Vector4d& q, const Eigen::Vector3d& v, const Eigen::Vector3d& w, const Eigen::Vector3d& a,
                             Eigen::Vector4d& q_dot, Eigen::Vector3d& p_dot, Eigen::Vector3d& v_dot) {
        q_dot = 0.5 * Omega(w) * q;
        p_dot = v;
        Eigen::Matrix3d R = quat_2_Rot(quat_multiply(q, q_0));
        v_dot = R.transpose() * a - gravity;
    };

    Eigen::Vector4d k1_q_dot; Eigen::Vector3d k1_p_dot, k1_v_dot;
    compute_deriv(dq_0, v_0, w_hat1, a_hat1, k1_q_dot, k1_p_dot, k1_v_dot);
    Eigen::Vector4d k1_q = k1_q_dot * dt; Eigen::Vector3d k1_p = k1_p_dot * dt, k1_v = k1_v_dot * dt;

    Eigen::Vector3d w_mid = w_hat1 + 0.5 * w_alpha * dt;
    Eigen::Vector3d a_mid = a_hat1 + 0.5 * a_jerk * dt;

    Eigen::Vector4d dq_1 = (dq_0 + 0.5 * k1_q).normalized();
    Eigen::Vector3d v_1 = v_0 + 0.5 * k1_v;
    Eigen::Vector4d k2_q_dot; Eigen::Vector3d k2_p_dot, k2_v_dot;
    compute_deriv(dq_1, v_1, w_mid, a_mid, k2_q_dot, k2_p_dot, k2_v_dot);
    Eigen::Vector4d k2_q = k2_q_dot * dt; Eigen::Vector3d k2_p = k2_p_dot * dt, k2_v = k2_v_dot * dt;

    Eigen::Vector4d dq_2 = (dq_0 + 0.5 * k2_q).normalized();
    Eigen::Vector3d v_2 = v_0 + 0.5 * k2_v;
    Eigen::Vector4d k3_q_dot; Eigen::Vector3d k3_p_dot, k3_v_dot;
    compute_deriv(dq_2, v_2, w_mid, a_mid, k3_q_dot, k3_p_dot, k3_v_dot);
    Eigen::Vector4d k3_q = k3_q_dot * dt; Eigen::Vector3d k3_p = k3_p_dot * dt, k3_v = k3_v_dot * dt;

    Eigen::Vector4d dq_3 = (dq_0 + k3_q).normalized();
    Eigen::Vector3d v_3 = v_0 + k3_v;
    Eigen::Vector4d k4_q_dot; Eigen::Vector3d k4_p_dot, k4_v_dot;
    compute_deriv(dq_3, v_3, w_hat2, a_hat2, k4_q_dot, k4_p_dot, k4_v_dot);
    Eigen::Vector4d k4_q = k4_q_dot * dt; Eigen::Vector3d k4_p = k4_p_dot * dt, k4_v = k4_v_dot * dt;

    Eigen::Vector4d dq = (dq_0 + (1.0 / 6.0) * k1_q + (1.0 / 3.0) * k2_q + (1.0 / 3.0) * k3_q + (1.0 / 6.0) * k4_q).normalized();
    new_q = quat_multiply(dq, q_0);
    new_p = p_0 + (1.0 / 6.0) * k1_p + (1.0 / 3.0) * k2_p + (1.0 / 3.0) * k3_p + (1.0 / 6.0) * k4_p;
    new_v = v_0 + (1.0 / 6.0) * k1_v + (1.0 / 3.0) * k2_v + (1.0 / 3.0) * k3_v + (1.0 / 6.0) * k4_v;
}

// Perform clone augmentation: zero-allocation, runtime block() calls into pre-sized matrix
void augment_clone(State& state) {
    if (state.num_clones >= 12) { state.num_clones = 11; return; }
    int clone_idx = state.num_clones;
    state.num_clones++;

    Variable& new_clone = state.clones[clone_idx];
    new_clone.type = VariableType::PoseJPL;
    new_clone.id = 15 + clone_idx * 6;
    new_clone.size = 6;
    new_clone.storage_size = 7;
    new_clone.value[3] = 1.0;

    int s = new_clone.id;
    int total = 15 + state.num_clones * 6;
    // runtime block() calls — no static template sizes, no stack blowup
    state.Cov.block(s, 0, 6, total).setZero();
    state.Cov.block(0, s, total, 6).setZero();
    state.Cov.block(s, s, 6, 6).setIdentity();
}

// EKF Update in DOD (pre-allocated covariance slice, dynamic Jacobians on stack with bounded size)
void update_msckf(State& state) {
    int state_sz = 15 + 6 * state.num_clones;
    constexpr int MEAS = 22;

    // Use dynamic Eigen matrices — storage is on heap but allocation happens ONCE per call
    // (as opposed to Open_VINS which re-allocates covariance on every clone augmentation)
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(MEAS, state_sz);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(MEAS, MEAS) * 1e-3;

    // Operate on a block view into the pre-allocated covariance — zero copy
    auto Cov_active = state.Cov.block(0, 0, state_sz, state_sz);

    Eigen::MatrixXd S = H * Cov_active * H.transpose() + R;
    Eigen::MatrixXd K = Cov_active * H.transpose() * S.inverse();
    Cov_active -= K * H * Cov_active;
}

} // namespace our_style

// ============================================================================
// MAIN BENCHMARK LOOP
// ============================================================================
int main() {
    std::cout << "Benchmarking Open_VINS (Polymorphic/Resizing) vs Ported (Data-Oriented C++17)...\n";
    
    constexpr int NUM_ITERATIONS = 50000;
    
    Eigen::Vector3d w1(0.1, -0.2, 0.3);
    Eigen::Vector3d a1(0.01, -0.02, 9.80);
    Eigen::Vector3d w2(0.12, -0.18, 0.28);
    Eigen::Vector3d a2(0.02, -0.01, 9.82);
    double dt = 0.01;
    
    // 1. Open_VINS Style
    {
        openvins_style::State state;
        Eigen::Vector4d nq; Eigen::Vector3d nv, np;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            openvins_style::predict_mean_rk4(state, dt, w1, a1, w2, a2, nq, nv, np);
            
            Eigen::Matrix<double, 16, 1> update_val;
            update_val.head<4>() = nq;
            update_val.segment<3>(4) = np;
            update_val.segment<3>(7) = nv;
            update_val.segment<6>(10).setZero();
            state.imu->set_value(update_val);
            
            // Augment camera clone
            if (i % 5 == 0) {
                openvins_style::augment_clone(state);
            }
            
            // MSCKF visual update
            if (i % 10 == 0) {
                openvins_style::update_msckf(state);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "[OpenVINS style]: " << NUM_ITERATIONS << " runs took: " << diff.count() << " seconds (" 
                  << (diff.count() / NUM_ITERATIONS) * 1e6 << " ns/run)\n";
    }
    
    // 2. Our Ported DOD Style
    {
        our_style::State state;
        Eigen::Vector4d nq; Eigen::Vector3d nv, np;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            our_style::predict_mean_rk4(state, dt, w1, a1, w2, a2, nq, nv, np);
            
            state.imu.value[0] = nq(0);
            state.imu.value[1] = nq(1);
            state.imu.value[2] = nq(2);
            state.imu.value[3] = nq(3);
            state.imu.value[4] = np(0);
            state.imu.value[5] = np(1);
            state.imu.value[6] = np(2);
            state.imu.value[7] = nv(0);
            state.imu.value[8] = nv(1);
            state.imu.value[9] = nv(2);
            
            // Augment camera clone
            if (i % 5 == 0) {
                our_style::augment_clone(state);
            }
            
            // MSCKF visual update
            if (i % 10 == 0) {
                our_style::update_msckf(state);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "[Ported DOD style]: " << NUM_ITERATIONS << " runs took: " << diff.count() << " seconds (" 
                  << (diff.count() / NUM_ITERATIONS) * 1e6 << " ns/run)\n";
    }
    
    return 0;
}
