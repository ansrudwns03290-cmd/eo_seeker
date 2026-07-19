#include "state_estimation/StateEstimator.hpp"

StateEstimator::StateEstimator()
    : m_is_initialized(false),
      // [튜닝] 기존 1e-2는 predict()에서 dt*dt(30fps 기준 ≈0.0011)까지 곱해지고 나면
      // 실제 Q가 1e-5 수준까지 작아져, 측정 노이즈 R(1e-1)과 4~5자리 차이가 나버린다.
      // 그 결과 칼만 이득이 거의 0에 수렴해 "예측(노란 점)"과 "보정(파란 점)"이 거의
      // 겹쳐 보이고, 표적이 빠르게/갑자기 움직이면 필터가 한참 뒤에서 쫓아오는 현상이 생겼다.
      // Q와 R이 같은 자릿수대(0.01~0.1)가 되도록 계수를 대폭 올려서 필터가 새 측정값을
      // 더 적극적으로 반영하도록 한다. 다만 너무 키우면 KCF 자체의 프레임 단위 흔들림에도
      // 민감해져 박스가 떨릴 수 있으니, 실제 영상으로 재테스트하며 필요시 더 조정할 것
      // (반응성 부족하면 더 키우고, 대신 떨림이 심해지면 조금 낮추는 식으로 경험적으로 맞춘다).
      // 5.0에서 한 단계 더 올림 (실측 결과 여전히 약간 뒤처져서 반응성을 더 높임).
      // 계속 뒤처지면 더 올리고, 대신 박스가 눈에 띄게 떨리기 시작하면 이 값을 낮출 것.
      m_process_noise_coef(15.0f),
      m_measure_noise_coef(1e-1f)   // KCF 측정치가 흔들릴수록(노이즈가 심할수록) 이 값을 키웁니다.
{
    // 상태 벡터(4차원: x, y, vx, vy), 측정 벡터(2차원: x, y), 제어 벡터(0차원)
    m_kf.init(4, 2, 0);
}

void StateEstimator::initialize(const cv::Point2f& initial_pos)
{
    // 1. 상태 벡터 초기화 [x, y, vx, vy]^T
    m_kf.statePost.at<float>(0) = initial_pos.x;
    m_kf.statePost.at<float>(1) = initial_pos.y;
    m_kf.statePost.at<float>(2) = 0.0f; // 초기 속도는 0으로 가정
    m_kf.statePost.at<float>(3) = 0.0f;

    m_kf.statePre.at<float>(0) = initial_pos.x;
    m_kf.statePre.at<float>(1) = initial_pos.y;
    m_kf.statePre.at<float>(2) = 0.0f;
    m_kf.statePre.at<float>(3) = 0.0f;

    // 2. 전이 행렬 (Transition Matrix, A) 초기화 - 등속도 모델
    // [ 1  0  dt 0  ]
    // [ 0  1  0  dt ]
    // [ 0  0  1  0  ]
    // [ 0  0  0  1  ]
    // * dt는 매 predict 마다 가변적으로 바뀔 수 있으므로 여기서는 identity 기반 프레임만 잡습니다.
    cv::setIdentity(m_kf.transitionMatrix);

    // 3. 측정 행렬 (Measurement Matrix, H) 초기화
    // 상태 벡터 [x, y, vx, vy]에서 [x, y]만 측정하므로 시스템 매핑을 해줍니다.
    // [ 1  0  0  0 ]
    // [ 0  1  0  0 ]
    m_kf.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
    m_kf.measurementMatrix.at<float>(0, 0) = 1.0f;
    m_kf.measurementMatrix.at<float>(1, 1) = 1.0f;

    // 4. 측정 노이즈 공분산 행렬 (Measurement Noise Covariance, R)
    cv::setIdentity(m_kf.measurementNoiseCov, cv::Scalar::all(m_measure_noise_coef));

    // 5. 오차 공분산 행렬 (Error Covariance, P) 초기화
    cv::setIdentity(m_kf.errorCovPost, cv::Scalar::all(1.0f));

    m_is_initialized = true;
}

cv::Point2f StateEstimator::predict(double dt)
{
    if (!m_is_initialized) return cv::Point2f(0, 0);

    // 매 프레임 변하는 dt를 전이 행렬(A)에 실시간으로 반영합니다.
    m_kf.transitionMatrix.at<float>(0, 2) = static_cast<float>(dt);
    m_kf.transitionMatrix.at<float>(1, 3) = static_cast<float>(dt);

    // dt^2를 곱하여 시간에 따른 불확실성 증가 반영
    cv::setIdentity(m_kf.processNoiseCov, cv::Scalar::all(m_process_noise_coef * dt * dt)); 

    // OpenCV 내부 Kalman Predict 수행
    cv::Mat prediction = m_kf.predict();
    
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

cv::Point2f StateEstimator::update(const cv::Point2f& measurement_pos)
{
    if (!m_is_initialized) return cv::Point2f(0, 0);

    // 2차원 측정값 행렬 생성
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << measurement_pos.x, measurement_pos.y);

    // OpenCV 내부 Kalman Correct(Update) 수행
    cv::Mat estimated = m_kf.correct(measurement);

    return cv::Point2f(estimated.at<float>(0), estimated.at<float>(1));
}

cv::Point2f StateEstimator::getEstimatedPosition() const
{
    return cv::Point2f(m_kf.statePost.at<float>(0), m_kf.statePost.at<float>(1));
}

cv::Point2f StateEstimator::getEstimatedVelocity() const
{
    return cv::Point2f(m_kf.statePost.at<float>(2), m_kf.statePost.at<float>(3));
}

void StateEstimator::reset()
{
    m_is_initialized = false;
}