//
// Created by aaron on 5/1/18.
//

#ifndef SLITHERLINKASSISTANT_NATIVE_LIB_HPP
#define SLITHERLINKASSISTANT_NATIVE_LIB_HPP

#include <opencv2/core/mat.hpp>
#include <set>

struct MyPoint2fYComp {
    bool operator()(const cv::Point2f& left, const cv::Point2f& right) const;
};

struct MyPoint2fXComp {
    bool operator()(const cv::Point2f& left, const cv::Point2f& right) const;
};

typedef std::set<cv::Point2f, MyPoint2fYComp> PointSet;

/** This is where to make changes if we're missing too many gridpoints
 *
 * @param img The image in which to detect keypoints
 * @param kps A vector for storing detected keypoints
 */
void detectKeypoints(const cv::Mat& img, std::vector<cv::KeyPoint>& kps);

/**
 *
 * @param kps vector of keypoints from detectKeypoints
 * @param grid vector for storing confirmed grid points
 * @return the average distance between neighboring grid points
 */
double findStrongGridPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& grid);

double myDist(cv::Point2f left, cv::Point2f right);

void findNeighbors(const std::vector<cv::KeyPoint>& kps, cv::Point2f candidate,
                   std::vector<cv::Point2f>& neighbors, std::vector<double>& bestDists);

double calculateRotation(const std::vector<cv::Point2f>& grid);

void findWeakGridPoints(const std::vector<cv::KeyPoint>& kps, const std::vector<cv::Point2f>& strongGrid,
                        PointSet& gridSet, double angle, double unit);

void rotatePoint(cv::Point2f& pt, double angle);

std::pair<int, int> getPuzzleTopLeft(const cv::Mat& img, cv::Mat& binary, double thresh, double unit);

void completeGrid(std::vector<cv::Point2f>& grid, cv::Mat1f& gridXs, cv::Mat1f& gridYs,
                  std::pair<int, int> topLeft, int puzzleSize, double unit, double angle);

#endif //SLITHERLINKASSISTANT_NATIVE_LIB_HPP
