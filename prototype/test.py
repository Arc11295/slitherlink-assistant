import numpy as np
import skimage
import PIL.Image

from numpy import sin, cos, arctan
from numpy.linalg import lstsq
from matplotlib import pyplot as plt, cm
from skimage import feature as feat, filters as flt
from tesserocr import PyTessBaseAPI, PSM
from subprocess import Popen, PIPE

def my_dist(a1, a2, out):
    np.sum(np.power(a1-a2,2),1,out=out)
    np.sqrt(out, out)


def my_lin_fit(dep,ind):
    coeff = np.column_stack([ind,np.ones_like(ind)])
    return lstsq(coeff, dep)[0]


def find_puz_bound(img_edges, count_thresh, border_proportion=30):
    """Finds a bounding box around the slitherlink puzzle in the given image.
    """
    rows = img_edges.shape[0]
    cols = img_edges.shape[1]
    border = min(rows, cols)/border_proportion

    row_pixel_counts = np.sum(img_edges, 1)
    # np.where(condition) returns condition.nonzero(), which returns a
    # tuple of arrays, even when condition is 1-dimensional. This line has to
    # have [0][[0,-1]] at the end instead of just [[0,-1]] to extract
    # the index array from the tuple before getting the first and last elements
    top,bot = np.where(row_pixel_counts > count_thresh)[0][[0,-1]]
    # give border-many pixels of slack to make sure the whole puzzle is
    # below top and above bottom, but cut the border off at the image's edge
    top = max(0, top - border)
    bot = min(rows-1, bot + border)

    col_pixel_counts = np.sum(img_edges, 0)
    left,right = np.where(col_pixel_counts > count_thresh)[0][[0,-1]]
    left = max(0, left - border)
    right = min(cols-1, right + border)

    return top, bot, left, right


def process_candidates(candidates):
    ys,xs = candidates[:,0],candidates[:,1]
    # search for grid points with four neighbors
    curr_cand = np.zeros(2)
    dists = np.zeros_like(candidates[:,0])
    foundh = []
    grid_pts = []
    unit_dists = []
    for i in xrange(candidates.shape[0]): # each candidate is a row
        # fill curr_cand with values
        curr_cand[0] = ys[i]
        curr_cand[1] = xs[i]
        # compute distances between current candidate and all candidates
        my_dist(curr_cand, candidates, dists)
        # sort distances and get the best four
        best = np.argsort(dists)[1:5]

        # The distances between an actual grid point and its four neighbors
        # should have a low coefficient of variation among them.
        cv = np.std(dists[best])/np.mean(dists[best])
        if cv < 0.01:
            lowx,highx = min(xs[best]),max(xs[best])
            lowy,highy = min(ys[best]),max(ys[best])
            verify = [xs[i]-lowx, xs[i]-highx, ys[i]-lowy, ys[i]-highy]
            verify = map(abs, verify)
            cv2 = np.std(verify)/np.mean(verify)
            if cv2 < 0.01:
                samey = (xs[best] == highx) | (xs[best] == lowx)
                samex = (ys[best] == highy) | (ys[best] == lowy)

                h_fit_pts = np.vstack([candidates[best][samey], candidates[i]])
                _,b = my_lin_fit(h_fit_pts[:,0],h_fit_pts[:,1])
                h_fit_pts[:,0] -= b
                foundh.append(h_fit_pts)

                grid_pts.extend([candidates[i],
                     candidates[best][samey],
                     candidates[best][samex]
                    ])

                unit_dists.extend(list(dists[best]))

    hy,hx = np.hsplit(np.vstack(foundh),2)
    mh,_ = my_lin_fit(hy,hx)
    mh = mh[0] # for some reason my_lin_fit isn't returning mh as a scalar...
    # arctan(mh) is the angle the image is rotated, but we want to reverse
    # the rotation.
    t = -arctan(mh)

    return t, np.mean(unit_dists), list(np.vstack(grid_pts))


def complete_grid(grid, dimension, unit, t):
    grid = np.array(list(grid))
    # sort by y-coordinate
    ys_sorted = np.argsort(grid[:,0])
    grid[:,0] = grid[:,0][ys_sorted]
    grid[:,1] = grid[:,1][ys_sorted]
    # find the indices where there are significant changes in the y-coordinate
    row_bounds = [0]
    rotation = np.asarray([[cos(t), sin(t)],[-sin(t), cos(t)]])
    for i in xrange(grid.shape[0]-1):
        point = grid[i+1] - grid[i]
        point = np.matmul(rotation, point)
        if abs(point[0]) > unit/2:
            row_bounds.append(i+1)

    row_bounds.append(grid.shape[0])

    # the changes we found indicate that points before belong to one row of
    # the grid and points after belong to the next. Here we use this information
    # to sort each row of grid points by their x-coordinates
    for i,start in enumerate(row_bounds[:-1]):
        end = row_bounds[i+1]
        xs_sorted = np.argsort(grid[start:end,1])
        grid[start:end,0] = grid[start:end,0][xs_sorted]
        grid[start:end,1] = grid[start:end,1][xs_sorted]

    grid_xs = np.zeros((dimension+1, dimension+1),dtype=int)
    grid_ys = np.zeros_like(grid_xs)

    origin = grid[0]
    for i in xrange(grid.shape[0]):
        # translate
        point = grid[i] - origin
        # rotate and normalize
        point = np.matmul(rotation, point)/unit
        point = np.rint(point)
        grid_xs[int(point[0]),int(point[1])] = grid[i,1]
        grid_ys[int(point[0]),int(point[1])] = grid[i,0]

    # This might happen if blob detection misses some of the grid points
    # entirely, or if the image is distorted enough that some grid points do
    # not lie on a grid with any "strong" point as the origin. We want to
    # try to fill in the holes in this case.
    if grid.shape[0] < (dimension+1)**2:
        # if there are places where only one of x and y is zero, that's NOT a
        # hole. the only time we'll have something that looks like a hole but is
        # not is when the upper-leftmost grid point is at image coordinates
        # (0,0). In that case, though, it's also the origin we're using, so it'll
        # still map exactly correctly.
        holes = np.where((grid_xs == 0) & (grid_ys == 0))
        fill = np.vstack(holes)*unit
        rev_rotation = np.asarray([[cos(-t), sin(-t)],[-sin(-t), cos(-t)]])
        fill = np.matmul(rev_rotation, fill)
        fill += origin.reshape((1,2)).T
        fill = np.rint(fill)
        grid_xs[holes] = fill[1]
        grid_ys[holes] = fill[0]

    return grid_ys,grid_xs


def find_grid(blobs, use, dimension):
    """Finds the location of the slitherlink grid using detected blobs
    """
    candidates = blobs[use,:2]
    t, unit, grid_pts = process_candidates(candidates)

    # The sign of the sine terms is reversed compared to a typical
    # rotation matrix because the points both before and after
    # transformation should be (y,x), not (x,y)
    rotation = np.asarray([[cos(t), sin(t)],[-sin(t), cos(t)]])
    real_grid = {tuple(pt) for pt in grid_pts}

    others = blobs[~use,:2]
    for pt in list(candidates)+list(others):
        if tuple(pt) in real_grid:
            continue
        for origin in grid_pts:
            pt2 = pt - origin
            pt2 = np.matmul(rotation, pt2)
            pt2 /= unit
            err = abs(np.rint(pt2) - pt2)
            if err[0] < 0.1 and err[1] < 0.1:
                real_grid.add(tuple(pt))
                break

    ys,xs = complete_grid(real_grid, dimension, unit, t)
    return ys,xs,grid_pts


def find_numbers(img, ys, xs, confidence_thresh=70, cell_crop=10):
    with PyTessBaseAPI(psm=PSM.SINGLE_CHAR) as api:
        api.SetVariable('classify_bln_numeric_mode', '1')
        grid = []
        for i in xrange(xs.shape[0]-1):
            for j in xrange(xs.shape[1]-1):
                # These bounds are basically as tight as possible while
                # guaranteeing that they won't cut off part of any numbers.
                top = np.maximum(ys[i,j], ys[i,j+1])+cell_crop
                left = np.maximum(xs[i,j], xs[i+1,j])+cell_crop
                right = np.minimum(xs[i,j+1], xs[i+1,j+1])-cell_crop
                bot = np.minimum(ys[i+1,j], ys[i+1,j+1])-cell_crop
                cell = PIL.Image.fromarray(img[top:bot, left:right])
                api.SetImage(cell)
                detected = api.GetUTF8Text().strip()
                try:
                    confidence = api.AllWordConfidences()[0]
                except IndexError:
                    confidence = 0
                cell_contents = '.'
                if confidence >= confidence_thresh:
                    try:
                        int(detected)
                    except ValueError:
                        pass
                    else:
                        cell_contents = detected
                grid.append(cell_contents)

            grid.append('\n')
        return "".join(grid)


def main():
    low_sigma = 1
    high_sigma = 7.5
    puzzle_dim = 6
    # puzzle_rows = 7
    # puzzle_cols = 7
    # TODO: lower this target and try to find parameters that work. The more the
    # image can be scaled down, the faster blob detection will be.
    target_pix_per_block = 141.5
    I = plt.imread('p1_s1.tiff')
    # TODO: find image boundary without finding edges first. This will eliminate
    # a pretty significant bottleneck. We can always detect edges after rescaling
    # in order to find the lines between grid points
    I_edges = feat.canny(I, sigma=2)
    top, bot, left, right = find_puz_bound(I_edges, 60)
    # bot and right are supposed to be included
    I_crop = I[top:bot+1,left:right+1]
    pix_per_block = I_crop.shape[1]/puzzle_dim
    rescale_factor = target_pix_per_block/pix_per_block
    I = flt.gaussian(I, 1/rescale_factor)
    I = skimage.transform.rescale(I, rescale_factor, order=3)
    I *= 255
    I = np.rint(I)
    I = np.asarray(I, dtype=np.uint8)
    blobs = feat.blob_doh(I,min_sigma=low_sigma,max_sigma=high_sigma)
    # use = (blobs[:,2] > 16) & (blobs[:,2] < 18)
    print np.unique(blobs[:,2])
    use = (blobs[:,2] > 6.8) & (blobs[:,2] < 9)
    # real_grid = np.array(list(find_grid(blobs,use)))
    ys,xs,strong_pts = find_grid(blobs,use,puzzle_dim)
    strongys,strongxs = np.hsplit(np.vstack(strong_pts),2)
    grid_blobs = blobs[use,:]

    numbers = find_numbers(I, ys, xs)
    print numbers
    p = Popen(['./solve'], stdin=PIPE, stdout=PIPE)
    solution = p.communicate(numbers)[0]
    print "got solution! here it is:"
    print solution

    # TODO: detect solution in image and compare to correct solution
    # consider using ajacency list to represent where lines are drawn

    fig1 = plt.figure()
    plt.imshow(I,cmap=cm.gray)
    # rows = I.shape[0]
    # cols = I.shape[1]
    # xs = np.array([[0,0,left,right],[cols,cols,left,right]])
    # ys = np.array([[top,bot,0,0],[top,bot,rows,rows]])
    # plt.plot(xs, ys)
    plt.scatter(blobs[:,1], blobs[:,0],cmap=cm.jet, c=blobs[:,2],edgecolors='none')
    plt.colorbar()

    fig2 = plt.figure()
    plt.imshow(I,cmap=cm.gray)
    plt.scatter(grid_blobs[:,1],grid_blobs[:,0],cmap=cm.jet,c=grid_blobs[:,2],edgecolors='none')
    plt.colorbar()

    fig3 = plt.figure()
    plt.imshow(I,cmap=cm.gray)
    plt.scatter(grid_blobs[:,1],grid_blobs[:,0],cmap=cm.jet,c=grid_blobs[:,2],edgecolors='none',s=70)
    plt.colorbar()
    plt.scatter(strongxs[::5], strongys[::5],c='r',s=50,edgecolors='none')
    plt.scatter(strongxs[1::5], strongys[1::5],c='b',s=30, edgecolors='none')
    plt.scatter(strongxs[2::5], strongys[2::5],c='b',s=30, edgecolors='none')
    plt.scatter(strongxs[3::5], strongys[3::5],c='y',s=10, edgecolors='none')
    plt.scatter(strongxs[4::5], strongys[4::5],c='y',s=10, edgecolors='none')

    fig4 = plt.figure()
    plt.imshow(I,cmap=cm.gray)
    plt.scatter(np.ravel(xs),np.ravel(ys),c='r',s=40,edgecolors='none')

    plt.show()

if __name__ == "__main__":
    main()
