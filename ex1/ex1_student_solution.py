"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        col_number = match_p_dst.shape[1]
        essential_m = np.zeros((2 * col_number, 9))
        for i in range(col_number):
            src_p = match_p_src[:, i]
            dst_p = match_p_dst[:, i]
            essential_m[i * 2] = [src_p[0], src_p[1], 1, 0, 0, 0, -dst_p[0] * src_p[0], -dst_p[0] * src_p[1], -dst_p[0]]
            essential_m[i * 2 + 1] = [0, 0 ,0 ,src_p[0], src_p[1], 1, -dst_p[1] * src_p[0], -dst_p[1] * src_p[1], -dst_p[1]]

        
        U, S, Vh = np.linalg.svd(essential_m)
        eigen_vec = Vh[-1]
        homography = eigen_vec.reshape(3, 3)
        """
            TEST homography
            >>> res = np.matmul(homography, np.append(match_p_src[:,0], 1))
            >>> res = np.round(res / res[-1])[:-1]
            >>> print(res, match_p_dst[:,0])
        """
        return homography

    def approx_point(self, homography, src_point):
        res = np.matmul(homography, np.append(src_point, 1))
        return np.round(res / res[-1])[:-1]
    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        dst_image = np.ndarray(dst_image_shape)
        # return new_image
        for src_y in range(src_image.shape[0]):
            for src_x in range(src_image.shape[1]):
                src_loc = np.array([src_x, src_y, 1])
                dst_loc = np.matmul(homography, src_loc)
                dst_loc = dst_loc / dst_loc[2]
                dst_loc_x, dst_loc_y = int(dst_loc[0]), int(dst_loc[1])
                if dst_loc_x < 0 or dst_loc_y < 0:
                    continue
                dst_image[dst_loc_y, dst_loc_x] = src_image[src_y, src_x]
        dst_image = dst_image.astype(np.uint8)
        return dst_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        dst_image = np.zeros(dst_image_shape)
        rows = np.arange(src_image.shape[0])
        cols = np.arange(src_image.shape[1])

        # return new_image
        cols_mat, rows_mat = np.meshgrid(cols, rows)
        # Add ones for the homogenous coordinates
        stack = np.stack((cols_mat, rows_mat, np.ones_like(rows_mat)), axis=0)
        coordinates = stack.reshape(3, -1)
        dest_coordinates = np.matmul(homography, coordinates)
        # TEST: validate that (Pdb) [921., 409.] is mapped to [ 94., 400.]
        division = np.repeat(dest_coordinates[-1,:].reshape(1, -1), repeats=2 ,axis=0)
        dest_coordinates = dest_coordinates[:-1, :] / division
        dest_coordinates = np.round(dest_coordinates)
        dest_coordinates = dest_coordinates.reshape((-1, src_image.shape[0], src_image.shape[1]))

        # Clipping dest
        clipped_y_values = np.clip(dest_coordinates[1,:,:], a_min=0, a_max=dst_image_shape[0] - 1).astype(np.int32)
        clipped_x_values = np.clip(dest_coordinates[0,:,:], a_min=0, a_max=dst_image_shape[1] - 1).astype(np.int32)

        dst_image[clipped_y_values, clipped_x_values] = src_image[rows_mat, cols_mat]
        dst_image = dst_image.astype(np.uint8)
        return dst_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        homo_src_p = np.stack((match_p_src[0, :], match_p_src[1, :], np.ones_like(match_p_src[0,:])))
        approx_dest = np.matmul(homography, homo_src_p)

        division = np.repeat(approx_dest[-1,:].reshape(1, -1), repeats=2 ,axis=0)

        approx_dest = approx_dest[:-1, :] / division
        approx_dest = np.round(approx_dest)
        distance = np.sqrt((approx_dest[0, :] - match_p_dst[0, :]) ** 2 + (approx_dest[1, :] - match_p_dst[1, :]) ** 2)
        fit_percent = np.count_nonzero(distance <= max_err) / match_p_dst.shape[1]
        inlier_distance = distance[np.nonzero(distance <= max_err)]
        dist_mse = (inlier_distance ** 2).mean() 
        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        mp_src_meets_model = None
        mp_dst_meets_model = None
        for src_p, dst_p  in zip(match_p_src.T, match_p_dst.T):
            homo_src_p = np.array([src_p[0], src_p[1], 1])
            approx_dest =  np.matmul(homography, homo_src_p)
            approx_dest = approx_dest.reshape(-1, 1)
            division = np.repeat(approx_dest[-1,:].reshape(1, -1), repeats=2 ,axis=0)
            approx_dest = approx_dest[:-1, :] / division
            approx_dest = np.round(approx_dest)

            distance = int(np.sqrt((approx_dest[0, 0] - dst_p[0]) ** 2 + (approx_dest[1, 0] - dst_p[1]) ** 2))
            if distance > max_err:
                continue
            if mp_src_meets_model is None:
                mp_src_meets_model = np.expand_dims(src_p, axis=1)
                mp_dst_meets_model = np.expand_dims(dst_p, axis=1)
            else:
                mp_src_meets_model = np.insert(mp_src_meets_model, -1, src_p, axis=1)
                mp_dst_meets_model = np.insert(mp_dst_meets_model, -1, dst_p, axis=1)
        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5 #TODO - I lowered it down, I should undestand why it failed with 0.5
        # number of points sufficient to compute the model
        #TODO changed it to 8, for some reason the homography is not good when calculated with 4 points :(
        n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        best_homography = None
        max_meet_model_points = 0
        print(f"RANSAC running for {k} iterations")
        print(f"Minimal needed inliers {int(d * match_p_src.shape[1])}")
        
        for i in range(k):
            indices_choice = np.random.choice(np.arange(match_p_src.shape[1]) ,replace=False, size=(n))
            print(f"RANSAC iter {i} chose indices {indices_choice}")
            src_choice = match_p_src[:, indices_choice]
            dst_choice = match_p_dst[:, indices_choice]
            homography = self.compute_homography_naive(src_choice, dst_choice)
            meet_model_src, meet_model_dst = self.meet_the_model_points(homography, match_p_src, match_p_dst, t)
            #TODO how sometimes less than 4 points matches the homography...
            if meet_model_src is None:
                continue
            match_point_num = meet_model_src.shape[1]
            print(f"{match_point_num} points matched the homography")
            if match_point_num >= int(d * match_p_src.shape[1]):
                print(f"enough inliers - computing the model again")
                homography = self.compute_homography_naive(meet_model_src, meet_model_dst)
                meet_model_src, meet_model_dst = self.meet_the_model_points(homography, match_p_src, match_p_dst, t)
                if meet_model_src is None:
                    continue
                match_point_num = meet_model_src.shape[1]
                print(f"{match_point_num} After recalculation")
                if match_point_num > max_meet_model_points:
                    max_meet_model_points = meet_model_src.shape[1]
                    best_homography = homography
        if best_homography is None:
            raise ValueError("Could not find good homography oof")
        #TODO maybe we should scale it to 1?
        return best_homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        dst_rows = np.arange(dst_image_shape[0])
        dst_cols = np.arange(dst_image_shape[1])
        dst_cols_meshgrid, dest_rows_meshgrid = np.meshgrid(dst_cols, dst_rows) # (x, y)
        # Add ones for the homogenous coordinates
        homo_dest_coordinates = np.stack((dst_cols_meshgrid, dest_rows_meshgrid, np.ones_like(dest_rows_meshgrid)), axis=0) # (x, y, 1)
        homo_dest_coordinates = homo_dest_coordinates.reshape(3, -1)
        homo_approx_src_coordinates = np.matmul(backward_projective_homography, homo_dest_coordinates) # source = H * dest
        # TEST: ([1368,  414,    1]) -> [921., 409.]
        division = np.repeat(homo_approx_src_coordinates[-1,:].reshape(1, -1), repeats=2 ,axis=0)
        approx_src_coordinates = np.round(homo_approx_src_coordinates[:-1, :] / division)
        # Not rounding, we want the approx values on purpose
        approx_src_coordinates = approx_src_coordinates.reshape((-1, dst_image_shape[0], dst_image_shape[1])) # (2, x, y)

        src_rows = np.arange(src_image.shape[0])
        src_cols = np.arange(src_image.shape[1])

        src_cols_meshgrid, src_rows_meshgrid = np.meshgrid(src_cols, src_rows) # (x, y)
        # The griddata line doesnt work on my weak pc
        return griddata(points=(src_cols_meshgrid.flatten(), src_rows_meshgrid.flatten()), values=src_image.reshape((-1, 3)), xi=(approx_src_coordinates[0,:,:].flatten(), (approx_src_coordinates[1,:,:].flatten())), method="cubic").reshape((dst_image_shape[0], dst_image_shape[1], -1))


    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # I think it should with minus -pad_left because we need to transform the dest image to the left and than project it
        translation_matrix = np.array([[1, 0, -pad_left], [0, 1, -pad_up], [0, 0, 1]])
        final_homography =  np.matmul(backward_homography, translation_matrix)
        scale = np.linalg.norm(final_homography)
        return final_homography / scale

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        backward_homography = np.linalg.inv(homography)
        #TODO how can it be padded up and down?
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image, dst_image, homography)

        backward_homography_with_translation = self.add_translation_to_backward_homography(backward_homography, pad_struct.pad_left,  pad_struct.pad_up)
        panorama_image = np.ndarray((panorama_rows_num, panorama_cols_num, 3))

        dst_rows = np.arange(dst_image.shape[0])
        dst_cols = np.arange(dst_image.shape[1])

        # return new_image
        dst_rows_mat, dst_cols_mat = np.meshgrid(dst_rows, dst_cols)

        # This is a good line
        panorama_image[dst_rows_mat + pad_struct.pad_up, dst_cols_mat + pad_struct.pad_left] = dst_image[dst_rows_mat, dst_cols_mat]
        
        # Add backward mapping to the panorama if panorama is zero and backward mapping is not zero and not none
        backward_warped = self.compute_backward_mapping(backward_homography_with_translation, src_image, panorama_image.shape)

        backward_warped[backward_warped == np.NaN] = 0
        panorama_image[panorama_image == 0] = backward_warped[panorama_image == 0]
        return np.clip(panorama_image, 0, 255).astype(np.uint8)
        

