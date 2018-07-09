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

void cleanUpGlobals();

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

/**
 *
 * @param left
 * @param right
 * @return
 */
double myDist(cv::Point2f left, cv::Point2f right);

/**
 *
 * @param kps
 * @param candidate
 * @param neighbors
 * @param bestDists
 */
void findNeighbors(const std::vector<cv::KeyPoint>& kps, cv::Point2f candidate,
                   std::vector<cv::Point2f>& neighbors, std::vector<double>& bestDists);

/**
 *
 * @param grid
 * @return
 */
double calculateRotation(const std::vector<cv::Point2f>& grid);

/**
 *
 * @param kps
 * @param strongGrid
 * @param gridSet
 * @param angle
 * @param unit
 */
void findWeakGridPoints(const std::vector<cv::KeyPoint>& kps, const std::vector<cv::Point2f>& strongGrid,
                        PointSet& gridSet, double angle, double unit);

/**
 *
 * @param pt
 * @param angle
 */
void rotatePoint(cv::Point2f& pt, double angle);

/**
 *
 * @param img
 * @param binary
 * @param thresh
 * @param unit
 * @return
 */
std::pair<int, int> getPuzzleTopLeft(const cv::Mat& img, cv::Mat& binary, double thresh, double unit);

/**
 *
 * @param grid
 * @param gridXs
 * @param gridYs
 * @param topLeft
 * @param puzzleSize
 * @param unit
 * @param angle
 */
void completeGrid(std::vector<cv::Point2f>& grid, cv::Mat1i& gridXs, cv::Mat1i& gridYs,
                  std::pair<int, int> topLeft, int puzzleSize, double unit, double angle);

/**
 *
 * @param img
 * @param gridXs
 * @param gridYs
 * @param confThresh
 * @return
 */
std::string findNumbers(const cv::Mat& imgGray, cv::Mat& img, const cv::Mat1i& gridXs, const cv::Mat1i& gridYs,
                        const char* tessParent, int puzzleSize, int confThresh = 70, int cellCrop = 20);

/**
 *
 * @param img
 * @param text
 * @param box
 */
void putTextInBox(cv::Mat& img, const std::string& text, cv::Rect box);

std::string convertToLoopy(const std::string& puzzle, int puzzleSize);
#endif //SLITHERLINKASSISTANT_NATIVE_LIB_HPP
