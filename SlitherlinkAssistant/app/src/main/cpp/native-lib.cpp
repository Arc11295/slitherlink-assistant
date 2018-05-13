//
// Created by aaron on 4/17/18.
//
#include "native-lib.hpp"
#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <set>
#include <cmath>
#include <android/log.h>
#define LOG_TAG "native-lib"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)

using namespace cv;

extern "C" JNIEXPORT void
JNICALL
Java_com_github_arc11295_slitherlinkassistant_ProcessImageActivity_processImage(
        JNIEnv *env, jobject instance, jlong addr, jint puzzleSize) {
    Mat& myMat = *(Mat*) addr;
    Mat imgGray;
    cvtColor(myMat, imgGray, COLOR_RGB2GRAY);

    std::vector<KeyPoint> keypoints{};

    detectKeypoints(imgGray, keypoints);
    /*for (auto k : keypoints) {
        auto point = k.pt;
        circle(myMat, point, 15, Scalar(255, 0, 0), CV_FILLED);
    }*/

    std::vector<Point2f> grid{};
    double avgUnit = findStrongGridPoints(keypoints, grid);

    /*ALOGD("there are %d points in the grid", (int) grid.size());
    for (auto i = grid.begin(); i != grid.end(); ++i) {
        circle(myMat, *i, 10, Scalar(0, 0, 255), CV_FILLED);
    }*/

    double theta = calculateRotation(grid);

    PointSet gridSet{};

    findWeakGridPoints(keypoints, grid, gridSet, theta, avgUnit);

    /*for (auto p : gridSet) {
        circle(myMat, p, 5, Scalar(0, 255, 0), CV_FILLED);
    }*/

    grid = std::vector<Point2f>(gridSet.begin(), gridSet.end());
    Mat1f gridXs(puzzleSize+1, puzzleSize+1, 0.0);
    Mat1f gridYs(puzzleSize+1, puzzleSize+1, 0.0);
    Mat binary;
    //TODO make sure we're consistently getting the actual edges of the puzzle
    auto pair = getPuzzleTopLeft(imgGray, binary, 0.2, avgUnit);
    completeGrid(grid, gridXs, gridYs, pair, puzzleSize, avgUnit, theta);

    for (int i = 0; i < gridXs.rows; ++i) {
        for (int j = 0; j < gridXs.cols; ++j) {
            Point2f point{};
            point.x = gridXs(i,j);
            point.y = gridYs(i,j);
            circle(myMat, point, 10, Scalar(255,0,0), CV_FILLED);
        }
    }
}

void detectKeypoints(const Mat& img, std::vector<KeyPoint>& kps) {
    SimpleBlobDetector::Params params{};
    params.filterByColor = true;
    params.blobColor = 0;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;
    params.maxConvexity = 1.0;
    auto myDetector = SimpleBlobDetector::create(params);

    myDetector->detect(img, kps);
}

double findStrongGridPoints(const std::vector<KeyPoint>& kps, std::vector<Point2f>& grid) {
    double distTotal = 0;
    int numDists = 0;

    for (KeyPoint kp : kps) {
        Point2f candidate = kp.pt;
        ALOGD("candidate is at (%f, %f)", candidate.x, candidate.y);
        std::vector<Point2f> neighbors;
        std::vector<double> bestDists;
        findNeighbors(kps, candidate, neighbors, bestDists);
        if (neighbors.size() < 4) {
            ALOGD("DANGER DANGER WILL ROBINSON");
        }
        ALOGD("neighbors:");
        for (Point2f n : neighbors) {
            ALOGD("(%f, %f)", n.x, n.y);
        }
        //Use information about four neighbors to see if candidate is a grid point or not
        Scalar mean{};
        Scalar std{};
        meanStdDev(bestDists,mean,std);
        // the distances from candidate to its neighbors should have low coefficient of variation
        // if it's a real grid point
        //TODO make sure this threshold continues to work
        if (std[0]/mean[0] > 0.07) {
            ALOGD("distances to neighbors have too high variation");
            continue;
        }

        std::vector<double> neighborXs;
        std::vector<double> neighborYs;
        for (Point2f n : neighbors) {
            neighborXs.push_back(n.x);
            neighborYs.push_back(n.y);
        }
        double hiX, loX, hiY, loY;
        minMaxIdx(neighborXs, &loX, &hiX);
        minMaxIdx(neighborYs, &loY, &hiY);
        std::vector<double> verify = {abs(loX-candidate.x), abs(hiX-candidate.x),
                                      abs(loY-candidate.y), abs(hiY-candidate.y)};
        meanStdDev(verify, mean, std);
        //TODO make sure this threshold continues to work
        if (std[0]/mean[0] > 0.01) {
            ALOGD("neighbors are rotated different amounts");
            continue;
        }
        ALOGD("found an actual grid point");

        grid.push_back(candidate);
        for (Point2f n : neighbors) {
            grid.push_back(n);
        }

        for (double dist : bestDists) {
            distTotal += dist;
            ++numDists;
        }
    }

    ALOGD("about to return");
    return distTotal/numDists;
}

void findNeighbors(const std::vector<KeyPoint>& kps, Point2f candidate,
                   std::vector<Point2f>& neighbors, std::vector<double>& bestDists) {
    //compute pairwise distances, keeping track of the smallest four greater than 0
    for (KeyPoint kp : kps) {
        double dist = myDist(candidate, kp.pt);
        auto neighborIter = neighbors.begin();
        bool inserted = false;
        for (auto distIter = bestDists.begin(); distIter != bestDists.end(); ++distIter) {
            if (dist > 0 and dist < *distIter) {
                bestDists.insert(distIter, dist);
                neighbors.insert(neighborIter, kp.pt);
                if (bestDists.size() > 4) {
                    bestDists.pop_back();
                    neighbors.pop_back();
                }
                inserted = true;
                break;
            }
            ++neighborIter;
        }
        if (!inserted and bestDists.size() < 4) {
            neighbors.push_back(kp.pt);
            bestDists.push_back(dist);
        }
    }
}

double myDist(Point2f left, Point2f right) {
    return (sqrt(pow((left.x - right.x), 2) + pow((left.y - right.y), 2)));
}

double calculateRotation(const std::vector<Point2f>& grid) {
    assert(grid.size() % 5 == 0);
    //TODO change this assert into checking the grid size beforehand and informing the user of an error
    assert(grid.size() > 0);

    std::vector<Point2f> fitPoints{};
    for (auto i = grid.begin(); i != grid.end(); std::advance(i,5)) {
        auto left{i};
        auto right{i};
        ++left;
        std::advance(right,5);
        auto pair = std::minmax_element(left, right, [](Point2f a, Point2f b) {return a.x < b.x;});
        std::vector<Point2f> sameY{};
        sameY.push_back(*i);
        sameY.push_back(*pair.first);
        sameY.push_back(*pair.second);
        Vec4d line{};
        fitLine(sameY, line, DIST_L2, 0, 0.01, 0.01);
        double vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
        double adjust = x0/vx;
        // y intercept of the fit line
        double b = y0 - (vy * adjust);
        for (auto j = sameY.begin(); j != sameY.end(); ++j) {
            (*j).y -= b;
            fitPoints.push_back(*j);
        }
    }

    Vec4d line{};
    fitLine(fitPoints, line, DIST_L2, 0, 0.01, 0.01);
    // This result is negated so we can use it directly as the theta of a rotation matrix to correct
    // for the angle the image is rotated by
    return -atan2(line[1], line[0]);
}

void findWeakGridPoints(const std::vector<KeyPoint>& kps, const std::vector<Point2f>& strongGrid,
                        PointSet& gridSet, double angle, double unit) {
    for (Point2f pt : strongGrid) {
        gridSet.insert(pt);
    }
    ALOGD("successfully inserted strong grid points");
    for (KeyPoint kp : kps) {
        Point2f candidate = kp.pt;
        if (gridSet.find(candidate) != gridSet.end()) {
            continue;
        }

        for (Point2f origin : strongGrid) {
            Point2f cand2 = candidate - origin;
            rotatePoint(cand2, angle);
            cand2 /= unit;
            Point2f err = Point2i(cand2);
            assert(err != cand2);
            err -= cand2;
            // TODO make sure these thresholds continue to work
            if (abs(err.x) < 0.05 and abs(err.y) < 0.05) {
                gridSet.insert(candidate);
                break;
            }
        }
    }
}

bool MyPoint2fYComp::operator()(const Point2f &left, const Point2f &right) const {
    return left.y < right.y;
}

bool MyPoint2fXComp::operator()(const Point2f &left, const Point2f &right) const {
    return left.x < right.x;
}

void rotatePoint(Point2f& pt, double angle) {
    float tx = pt.x, ty = pt.y;
    pt.x = static_cast<float>(cos(angle) * tx - sin(angle) * ty);
    pt.y = static_cast<float>(sin(angle) * tx + cos(angle) * ty);
}

std::pair<int, int> getPuzzleTopLeft(const Mat& img, Mat& binary, double thresh, double unit) {
    adaptiveThreshold(img, binary, 1, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 39, 10);
    double border = unit/3;
    int top = binary.rows;
    for (int i = 0; i < binary.rows; ++i) {
        Mat row = binary.row(i);
        Scalar total = sum(row);
        if (total[0] > thresh*binary.cols) {
            top = max(0, (int) (i - border));
            break;
        }
    }

    int left = binary.cols;
    for (int j = 0; j < binary.cols; ++j) {
        Mat col = binary.col(j);
        Scalar total = sum(col);
        if (total[0] > thresh*binary.rows) {
            left = max(0, (int) (j-border));
            break;
        }
    }
    return std::make_pair(top, left);
}

void completeGrid(std::vector<Point2f>& grid, Mat1f& gridXs, Mat1f& gridYs,
                  std::pair<int,int> topLeft, int puzzleSize, double unit, double angle) {
    // Find iterators pointing to elements where a new row of the grid begins
    std::vector<std::vector<Point2f>::iterator> rowBounds = {grid.begin()};
    auto gridLast = grid.end();
    --gridLast;
    for (auto i = grid.begin(); i != gridLast; ++i) {
        auto j{i};
        ++j;
        Point2f point = (*j) - (*i);
        rotatePoint(point, angle);
        if (abs(point.y) > unit/2) {
            rowBounds.push_back(j);
        }
    }
    rowBounds.push_back(grid.end());

    // Sort each row by x coordinates
    auto rowLast = rowBounds.end();
    --rowLast;
    for (auto start = rowBounds.begin(); start != rowLast; ++start) {
        auto stop{start};
        ++stop;
        std::sort(*start, *stop, MyPoint2fXComp{});
    }

    // Use the first grid point we have as the origin of a temporary image coordinate system
    Point2f imgOrigin = grid[0];
    int top = topLeft.first, left = topLeft.second;
    ALOGD("top is %d, left is %d", top, left);
    // Find where the temporary origin is in the grid coordinate system relative to the true origin
    Point2i gridOrigin = Point2d((imgOrigin.x - left)/unit, (imgOrigin.y - top)/unit);
    ALOGD("origin in img coords is at (%f, %f)", imgOrigin.x, imgOrigin.y);
    ALOGD("origin in grid coords is at (%d, %d)", gridOrigin.x, gridOrigin.y);
    for (Point2f pt : grid) {
        Point2f normCoords = pt - imgOrigin;
        rotatePoint(normCoords, angle);
        normCoords /= unit;
        Point2i gridCoords = normCoords;
        gridCoords += gridOrigin;
        gridXs(gridCoords.y, gridCoords.x) = pt.x;
        gridYs(gridCoords.y, gridCoords.x) = pt.y;
    }

    if (grid.size() < pow(puzzleSize + 1, 2)) {
        assert(gridXs.rows == gridYs.rows);
        assert(gridXs.cols == gridYs.cols);
        for (int i = 0; i < gridXs.rows; ++i) {
            for (int j = 0; j < gridXs.cols; ++j) {
                float x = j - gridOrigin.x;
                float y = i - gridOrigin.y;
                Point2f hole(x, y);
                hole *= unit;
                rotatePoint(hole, -angle);
                hole += imgOrigin;
                gridXs(i, j) = hole.x;
                gridYs(i, j) = hole.y;
            }
        }
    }
}
