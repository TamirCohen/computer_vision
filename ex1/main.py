import time
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from cv2 import resize, INTER_CUBIC
from matplotlib.patches import Circle

from ex1_student_solution import Solution


##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = '123456789'
ID2 = '987654321'
##########################################################


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


def load_data(is_perfect_matches=True):
    # Read the data:
    src_img = mpimg.imread('src.jpg')
    dst_img = mpimg.imread('dst.jpg')
    if is_perfect_matches:
        # loading perfect matches
        matches = scipy.io.loadmat('matches_perfect')
    else:
        # matching points and some outliers
        matches = scipy.io.loadmat('matches')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    return src_img, dst_img, match_p_src, match_p_dst


def main():
    solution = Solution()
    # Parameters
    max_err = 25
    inliers_percent = 0.8
    # loading data with perfect matches
    src_img, dst_img, match_p_src, match_p_dst = load_data()
    # Compute naive homography
    tt = time.time()
    naive_homography = solution.compute_homography_naive(match_p_src,
                                                         match_p_dst)
    print('Naive Homography {:5.4f} sec'.format(toc(tt)))
    print(naive_homography)

    # Plot naive homography with forward mapping, slow implementation
    tt = time.time()
    transformed_image = solution.compute_forward_homography_slow(
        homography=naive_homography,
        src_image=src_img,
        dst_image_shape=dst_img.shape)

    print('Naive Homography Slow computation takes {:5.4f} sec'.format(toc(tt)))
    plt.figure()
    forward_panorama_slow_plot = plt.imshow(transformed_image)
    plt.title('Forward Homography Slow implementation')
    # plt.show()

    # Plot naive homography with forward mapping, fast implementation
    tt = time.time()
    transformed_image_fast = solution.compute_forward_homography_fast(
        homography=naive_homography,
        src_image=src_img,
        dst_image_shape=dst_img.shape)

    print('Naive Homography Fast computation takes {:5.4f} sec'.format(toc(tt)))
    plt.figure()
    forward_panorama_fast_plot = plt.imshow(transformed_image_fast)
    plt.title('Forward Homography Fast implementation')
    # plt.show()

    # loading data with imperfect matches
    src_img, dst_img, match_p_src, match_p_dst = load_data(False)

    # Compute naive homography
    tt = time.time()
    naive_homography = solution.compute_homography_naive(match_p_src,
                                                         match_p_dst)
    print('Naive Homography for imperfect matches {:5.4f} sec'.format(toc(tt)))
    print(naive_homography)

    # Plot naive homography with forward mapping, fast implementation for
    # imperfect matches
    tt = time.time()
    transformed_image_fast = solution.compute_forward_homography_fast(
        homography=naive_homography,
        src_image=src_img,
        dst_image_shape=dst_img.shape)

    print('Naive Homography Fast computation for imperfect matches takes '
          '{:5.4f} sec'.format(toc(tt)))
    plt.figure()
    forward_panorama_imperfect_matches_plot = plt.imshow(transformed_image_fast)
    plt.title('Forward Panorama imperfect matches')
    # plt.show()

    # Test naive homography
    tt = time.time()
    fit_percent, dist_mse = solution.test_homography(naive_homography,
                                                     match_p_src,
                                                     match_p_dst,
                                                     max_err)
    print('Naive Homography Test {:5.4f} sec'.format(toc(tt)))
    print([fit_percent, dist_mse])

    # Compute RANSAC homography
    tt = tic()
    ransac_homography = solution.compute_homography(match_p_src,
                                                    match_p_dst,
                                                    inliers_percent,
                                                    max_err)
    print('RANSAC Homography {:5.4f} sec'.format(toc(tt)))
    print(ransac_homography)

    # Test RANSAC homography
    tt = tic()
    fit_percent, dist_mse = solution.test_homography(ransac_homography,
                                                     match_p_src,
                                                     match_p_dst,
                                                     max_err)
    print('RANSAC Homography Test {:5.4f} sec'.format(toc(tt)))
    print([fit_percent, dist_mse])

    # Build panorama
    tt = tic()
    img_pan = solution.panorama(src_img,
                                dst_img,
                                match_p_src,
                                match_p_dst,
                                inliers_percent,
                                max_err)
    print('Panorama {:5.4f} sec'.format(toc(tt)))

    # Course panorama
    plt.figure()
    course_panorama_plot = plt.imshow(img_pan)
    plt.title('Great Panorama')
    # plt.show()
    plt.show()


def your_images_loader():
    src_img_test = mpimg.imread('src_test.jpg')
    dst_img_test = mpimg.imread('dst_test.jpg')

    DECIMATION_FACTOR = 5.0
    src_img_test = resize(src_img_test,
                          dsize=(int(src_img_test.shape[1]/DECIMATION_FACTOR),
                                 int(src_img_test.shape[0]/DECIMATION_FACTOR)),
                          interpolation=INTER_CUBIC)
    dst_img_test = resize(dst_img_test,
                          dsize=(int(dst_img_test.shape[1]/DECIMATION_FACTOR),
                                 int(dst_img_test.shape[0]/DECIMATION_FACTOR)),
                          interpolation=INTER_CUBIC)

    matches_test = scipy.io.loadmat('matches_test')

    match_p_dst = matches_test['match_p_dst'].astype(float)
    match_p_src = matches_test['match_p_src'].astype(float)

    match_p_dst /= DECIMATION_FACTOR
    match_p_src /= DECIMATION_FACTOR
    return src_img_test, dst_img_test, match_p_src, match_p_dst


def your_images_main():
    solution = Solution()
    # Student Files
    # first run "create_matching_points.py" with your own images to create a mat
    # file with the matching coordinates.
    max_err = 25  # <<<<< YOU MAY CHANGE THIS
    inliers_percent = 0.8  # <<<<< YOU MAY CHANGE THIS

    src_img_test, dst_img_test, match_p_src, match_p_dst = your_images_loader()
    homography = solution.compute_homography(match_p_src, match_p_dst,
                                             inliers_percent,
                                             max_err=25)
    img = solution.compute_forward_homography_fast(
        homography=homography,
        src_image=src_img_test,
        dst_image_shape=dst_img_test.shape)
    plt.figure()
    import numpy as np
    student_forward_warp_img = plt.imshow(img.astype(np.uint8))
    plt.title('Forward warp example')
    # plt.show()

    backward_homography = solution.compute_homography(match_p_dst, match_p_src,
                                                      inliers_percent,
                                                      max_err=25)
    img = solution.compute_backward_mapping(
        backward_projective_homography=backward_homography,
                                      src_image=src_img_test,
                                      dst_image_shape=dst_img_test.shape)
    plt.figure()
    import numpy as np
    student_backward_warp_img = plt.imshow(img.astype(np.uint8))
    plt.title('Backward warp example')
    # plt.show()

    # Build student panorama
    tt = tic()
    img_pan = solution.panorama(src_img_test, dst_img_test, match_p_src,
                                match_p_dst, inliers_percent, max_err)
    print('Student Panorama {:5.4f} sec'.format(toc(tt)))

    plt.figure()
    student_panorama = plt.imshow(img_pan)
    plt.title('Awesome Panorama')
    # plt.show()

    # Build reversed student panorama
    tt = tic()
    img_pan2 = solution.panorama(dst_img_test, src_img_test, match_p_dst,
                                 match_p_src, inliers_percent, max_err)
    print('Student Panorama {:5.4f} sec'.format(toc(tt)))

    plt.figure()
    reversed_student_panorama = plt.imshow(img_pan2)
    plt.title('Reversed Awesome Panorama')
    plt.show()


if __name__ == '__main__':
    main()
    your_images_main()
