//
// Created by aaron on 4/17/18.
//
#include "native-lib.hpp"
#include "solving_rules_4.h"
#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <baseapi.h>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <android/log.h>
#define LOG_TAG "native-lib"
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define DEBUG 1

using namespace cv;

std::string globalLoopy;
SlinkerGrid globalUserSolution;

// TODO change the return type of this function so we can return error codes
extern "C" JNIEXPORT void
JNICALL
Java_com_github_arc11295_slitherlinkassistant_ProcessImageActivity_detectPuzzle(
        JNIEnv *env, jobject instance, jlong imgAddr, jint puzzleSize, jstring jTessParent) {
    Mat& img = *(Mat*) imgAddr;
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_RGB2GRAY);

    std::vector<KeyPoint> keypoints{};

    detectKeypoints(imgGray, keypoints);
#if DEBUG
    for (auto k : keypoints) {
        auto point = k.pt;
        circle(img, point, 15, Scalar(0, 0, 0), CV_FILLED);
    }
#endif

    std::vector<Point2f> grid{};
    double avgUnit = findStrongGridPoints(keypoints, grid);

#if DEBUG
    ALOGD("there are %d points in the grid", (int) grid.size());
    for (auto i = grid.begin(); i != grid.end(); ++i) {
        circle(img, *i, 10, Scalar(0, 0, 255), CV_FILLED);
    }
#endif

    // TODO return an error code
    if (grid.size() == 0) {
        return;
    }

    double theta = calculateRotation(grid);

    PointSet gridSet{};

    findWeakGridPoints(keypoints, grid, gridSet, theta, avgUnit);

#if DEBUG
    for (auto p : gridSet) {
        circle(img, p, 5, Scalar(0, 255, 0), CV_FILLED);
    }
#endif

    grid = std::vector<Point2f>(gridSet.begin(), gridSet.end());
    Mat1i gridXs(puzzleSize+1, puzzleSize+1, 0);
    Mat1i gridYs(puzzleSize+1, puzzleSize+1, 0);
    Mat binary;
    //TODO make sure we're consistently getting the actual edges of the puzzle
    // usually this does work but at least once it gave something like 700 and 200 for an image that
    // was pretty well aligned with the crop box, if perhaps off-center. probably worth investigating
    auto pair = getPuzzleTopLeft(imgGray, binary, 0.2, avgUnit);
    completeGrid(grid, gridXs, gridYs, pair, puzzleSize, avgUnit, theta);


    const char* tessParent = env->GetStringUTFChars(jTessParent, 0);

    std::string puzzle{findNumbers(imgGray, img, gridXs, gridYs, tessParent, puzzleSize)};

#if DEBUG
    for (int i = 0; i < gridXs.rows; ++i) {
        for (int j = 0; j < gridXs.cols; ++j) {
            Point2f point{};
            point.x = gridXs(i,j);
            point.y = gridYs(i,j);
            circle(img, point, 10, Scalar(255,0,0), CV_FILLED);
        }
    }

    ALOGD("Got the puzzle layout\n%s",puzzle.c_str());
#endif

    globalLoopy = convertToLoopy(puzzle, puzzleSize);
#if DEBUG
    ALOGD("loopy format string is %s", globalLoopy.c_str());
#endif

    globalUserSolution = SlinkerGrid::ReadFromLoopyFormat(globalLoopy);
    detectUserSolution(imgGray, img, gridXs, gridYs, globalUserSolution);

    env->ReleaseStringUTFChars(jTessParent, tessParent);
    //TODO: return a code for successful execution
}

extern "C" JNIEXPORT void
JNICALL
Java_com_github_arc11295_slitherlinkassistant_ProcessImageActivity_checkSolution(
        JNIEnv *env, jobject instance, jlong matAddr, jint puzzleSize) {
    // TODO: get correct solution from slinker, then compare to user solution

    SlinkerGrid sPuzzle = SlinkerGrid::ReadFromLoopyFormat(globalLoopy);
#if DEBUG
    ALOGD("gave the puzzle to slinker:\n%s", sPuzzle.GetPrintOut().c_str());
#endif
    sPuzzle.ClearBorders();
    auto rules = GetSolvingRules4();
    SlinkerGrid solution = sPuzzle.FindSolutions(rules, true, 1)[0];

#if DEBUG
    solution.MarkOffBordersAsUnknown();
    ALOGD("The solution is:\n%s", solution.GetPrintOut().c_str());
    solution.MarkUnknownBordersAsOff();
#endif
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
        std::vector<Point2f> neighbors;
        std::vector<double> bestDists;
        findNeighbors(kps, candidate, neighbors, bestDists);
#if DEBUG
        ALOGD("candidate is at (%f, %f)", candidate.x, candidate.y);
        if (neighbors.size() < 4) {
            ALOGD("DANGER DANGER WILL ROBINSON");
        }
        ALOGD("neighbors:");
        for (Point2f n : neighbors) {
            ALOGD("(%f, %f)", n.x, n.y);
        }
#endif
        //Use information about four neighbors to see if candidate is a grid point or not
        Scalar mean{};
        Scalar std{};
        meanStdDev(bestDists,mean,std);
        // the distances from candidate to its neighbors should have low coefficient of variation
        // if it's a real grid point
        //TODO make sure this threshold continues to work
        if (std[0]/mean[0] > 0.08) {
#if DEBUG
            ALOGD("distances to neighbors have too high variation");
#endif
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
#if DEBUG
            ALOGD("neighbors are rotated different amounts");
#endif
            continue;
        }
#if DEBUG
        ALOGD("found an actual grid point");
#endif

        grid.push_back(candidate);
        for (Point2f n : neighbors) {
            grid.push_back(n);
        }

        for (double dist : bestDists) {
            distTotal += dist;
            ++numDists;
        }
    }

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

// TODO change this function to return the PointSet by value, which will be efficient through move semantics
void findWeakGridPoints(const std::vector<KeyPoint>& kps, const std::vector<Point2f>& strongGrid,
                        PointSet& gridSet, double angle, double unit) {
    for (Point2f pt : strongGrid) {
        gridSet.insert(pt);
    }
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
            err -= cand2;
            // TODO make sure these thresholds continue to work
            if (abs(err.x) < 0.1 and abs(err.y) < 0.1) {
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

void completeGrid(std::vector<Point2f>& grid, Mat1i& gridXs, Mat1i& gridYs,
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

    //TODO split into two functions around this line: one to sort rows by x coordinate and the other to actually complete the grid
    // Use the first grid point we have as the origin of a temporary image coordinate system
    Point2f imgOrigin = grid[0];
    int top = topLeft.first, left = topLeft.second;
    // Find where the temporary origin is in the grid coordinate system relative to the true origin
    Point2i gridOrigin = Point2d((imgOrigin.x - left)/unit, (imgOrigin.y - top)/unit);
#if DEBUG
    ALOGD("top is %d, left is %d", top, left);
    ALOGD("origin in img coords is at (%f, %f)", imgOrigin.x, imgOrigin.y);
    ALOGD("origin in grid coords is at (%d, %d)", gridOrigin.x, gridOrigin.y);
#endif
    for (Point2f pt : grid) {
        Point2f normCoords = pt - imgOrigin;
        rotatePoint(normCoords, angle);
        normCoords /= unit;
        Point2i gridCoords = normCoords;
        gridCoords += gridOrigin;
        gridXs(gridCoords.y, gridCoords.x) = static_cast<int>(pt.x);
        gridYs(gridCoords.y, gridCoords.x) = static_cast<int>(pt.y);
    }

    if (grid.size() < pow(puzzleSize + 1, 2)) {
        assert(gridXs.rows == gridYs.rows);
        assert(gridXs.cols == gridYs.cols);
        for (int i = 0; i < gridXs.rows; ++i) {
            for (int j = 0; j < gridXs.cols; ++j) {
                if (gridXs(i,j) != 0 or gridYs(i,j) != 0) {
                    continue;
                }
                float x = j - gridOrigin.x;
                float y = i - gridOrigin.y;
                Point2f hole(x, y);
                hole *= unit;
                rotatePoint(hole, -angle);
                hole += imgOrigin;
                gridXs(i, j) = static_cast<int>(hole.x);
                gridYs(i, j) = static_cast<int>(hole.y);
            }
        }
    }
}

std::string findNumbers(const Mat& imgGray, Mat& img, const Mat1i& gridXs, const Mat1i& gridYs,
                        const char* tessParent, int puzzleSize, int confThresh, int cellCrop) {
    std::string puzzle;
    puzzle.reserve((size_t) pow(puzzleSize,2) + puzzleSize); //+ puzzlesize for end of line chars

    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
    if (api->Init(tessParent, "eng")) {
        ALOGE("Could not initialize tesseract.\n");
        exit(1);
    }
    api->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
    int width = imgGray.cols;
    int height = imgGray.rows;
    api->SetImage(imgGray.data, width, height, 1, width);
    api->SetVariable("classify_bln_numeric_mode", "1");
    for (int i = 0; i < gridXs.rows - 1; ++i) {
        for (int j = 0; j < gridXs.cols - 1; ++j) {
            int top = max(gridYs(i,j), gridYs(i,j+1)) + cellCrop;
            int left = max(gridXs(i,j), gridXs(i+1,j)) + cellCrop;
            int right = min(gridXs(i,j+1), gridXs(i+1,j+1)) - cellCrop;
            int bot = min(gridYs(i+1,j), gridYs(i+1,j+1)) - cellCrop;
            api->SetRectangle(left, top, right-left, bot-top);
            char* detected = api->GetUTF8Text();
            int* confidences = api->AllWordConfidences();
            int confidence = confidences[0];
            delete [] confidences;
#if DEBUG
            //rectangle(img, Rect(left, top, right-left, bot-top), Scalar(0,255,0), 3);
            ALOGD("detected: %s", detected);
            ALOGD("confidence: %d", confidence);
#endif
            if (confidence < 0 or confidence > 100) { //If confidence's value isn't valid
                confidence = 0; // Treat it as 0
                // This might happen e.g. if the api only detects space characters and so confidences
                // is a length-0 array (terminated with -1)
            }
            std::string contents = ".";
            if (confidence >= confThresh) {
#if DEBUG
                ALOGD("WEEEEE WOOOOOO WEEEEE WOOOOOOO WEEEEEE WOOOOOO WEEEE WOOOOOO");
#endif
                int value = -1;
                if (isdigit(detected[0])) {
                    value = atoi(detected);
                }
#if DEBUG
                ALOGD("got an integer value of %d", value);
#endif
                // The OCR engine should only detect numbers between 0 and 3 since that's all
                // that appears in slitherlink puzzles, but it's best to check to be safe
                if (value >= 0 and value < 4) {
                    contents = std::to_string(value);
                }
            }
            puzzle.append(contents);
            putTextInBox(img, contents, Rect(left, top, right-left, bot-top));
            delete [] detected;
        }
        puzzle.push_back('\n');
    }
    api->End();

    return puzzle;
}

void putTextInBox(Mat& img, const std::string& text, Rect box) {
    int thickness = 3;
    double scale = getFontScaleFromHeight(FONT_HERSHEY_SIMPLEX, box.height, thickness);
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, scale, thickness, NULL);
    // Note that Rect is defined by the x and y coordinates of its top-left corner, but the
    // putText function uses the BOTTOM-left corner
    Point boxCenter(box.x + (box.width/2), box.y + (box.height/2));
    Point org(boxCenter.x - (textSize.width/2), boxCenter.y + (textSize.height/2));
    putText(img, text, org, FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 255, 0), thickness);
}

std::string convertToLoopy(const std::string& puzzle, int puzzleSize) {
    std::stringstream loopy;
    loopy << puzzleSize << "x" << puzzleSize << ":";
    int n_spaces_pending = 0;
    for (auto i = puzzle.begin(); i != puzzle.end(); ++i) {
        switch(*i) {
            case '.':
                ++n_spaces_pending;
                break;
            case '\n':
                break;
            default:
                if (n_spaces_pending > 0) {
                    loopy << char('a'+n_spaces_pending-1);
                    n_spaces_pending = 0;
                }
                loopy << *i;
                break;
        }
    }
    if (n_spaces_pending > 0) {
        loopy << char('a'+n_spaces_pending-1);
    }

    return loopy.str();
}

void detectUserSolution(const Mat& imgGray, Mat& img, const Mat1i& gridXs,
                               const Mat1i& gridYs, SlinkerGrid userSolution) {
    for (int i = 0; i < gridXs.rows; ++i) {
        for (int j = 0; j < gridXs.cols; ++j) {
            Point2i current{gridXs(i,j), gridYs(i,j)};
            if (j+1 < gridXs.cols) {
                Point2i right{gridXs(i,j+1), gridYs(i,j+1)};
                int borderStatus = checkBorder(imgGray, current, right, HORIZONTAL);
                // These coordinates work out because slinker uses what amounts to an ascii-art
                // representation of the puzzle
                userSolution.gridValue((2*j)+1, 2*i) = borderStatus;
                drawBorder(img, current, right, borderStatus);
            }
            if (i+1 < gridXs.rows) {
                Point2i below{gridXs(i+1,j), gridYs(i+1,j)};
                int borderStatus = checkBorder(imgGray, current, below, VERTICAL);
                userSolution.gridValue(2*j, (2*i)+1) = borderStatus;
                drawBorder(img, current, below, borderStatus);
            }
        }
    }
}

int checkBorder(const Mat& imgGray, const Point2i& current, const Point2i& neighbor,
                BorderMode mode, int slackPixels) {
    int top, left, right, bot;
    if (mode == HORIZONTAL) {
        left = current.x;
        right = neighbor.x;
        top = min(current.y, neighbor.y) - slackPixels;
        bot = max(current.y, neighbor.y) + slackPixels;
    } else { //Vertical
        top = current.y;
        bot = neighbor.y;
        left = min(current.x, neighbor.x) - slackPixels;
        right = max(current.x, neighbor.x) + slackPixels;
    }

    Mat checkRegion(imgGray, Rect(left, top, right-left, bot-top));
    if (mode == HORIZONTAL) {
        // By transposing, we can turn the horizontal problem into a vertical problem, thus making
        // the rest of the function the same for both cases
        checkRegion = checkRegion.t();
    }

    Mat edges{checkRegion};
    // The datatype of these Mats should be 1 byte if I've done everything right, so the thresholds
    // I've used should be 10% and 20% of the max value possible here. I got 10% and 20% from the
    // default behavior of scikit-image's canny edge detector, which worked quite well in my prototype
    Canny(checkRegion, edges, 0.1d*UCHAR_MAX, 0.2d*UCHAR_MAX);
}

