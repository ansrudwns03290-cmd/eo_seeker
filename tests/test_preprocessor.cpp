#include <gtest/gtest.h>
#include "preprocessing/Preprocessor.hpp"

TEST(PreprocessorTest, ProcessFailsOnEmptyImage) {
    Preprocessor pre;
    Frame frame;  // image는 기본 생성된 빈 cv::Mat
    cv::Mat out;
    EXPECT_FALSE(pre.process(frame, out));
}

TEST(PreprocessorTest, ProcessConvertsToGrayscale) {
    Preprocessor pre;
    Frame frame;
    frame.image = cv::Mat(100, 150, CV_8UC3, cv::Scalar(10, 20, 30));
    frame.is_valid = true;

    cv::Mat out;
    ASSERT_TRUE(pre.process(frame, out));

    EXPECT_EQ(out.channels(), 1);
    EXPECT_EQ(out.rows, 100);
    EXPECT_EQ(out.cols, 150);
}
