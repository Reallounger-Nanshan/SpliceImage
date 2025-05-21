import os
import cv2
import numpy as np
from rich import print
 
class ImageSplicer():
    def __init__(self):
        # Get python file path
        file_path = os.path.dirname(__file__)
        project_path = file_path[:file_path.rfind("/") + 1]
        self.img_path = project_path + "img/"

        # Variable parameters
        self.imgs = ["5.jpg", "4.jpg", "1.jpg"]
        self.trees = 5
        self.tree_check_num = 50
        self.test_ratio = 0.65
        self.match_points_min = 10
        self.img_name = self.img_path + "img_splicing.jpg"
        self.show_matches = False

        # Constant value
        self.nearest_neighbor_num = 2

        # Create SIFT
        self.sift = cv2.SIFT_create()


    def MatchFeaturePoints(self, describe_left, describe_right):
            FLANN_INDEX_KDTREE = 1
            flann = cv2.FlannBasedMatcher({"algorithm": FLANN_INDEX_KDTREE, "trees": self.trees},
                                          {"checks": self.tree_check_num}
                                         )
            
            matches = flann.knnMatch(describe_left, describe_right, k = self.nearest_neighbor_num)

            return matches
    

    def CheckMatches(self, matches):
        good_matches = []
        
        for optimal_value, suboptimal_value in matches:
            if optimal_value.distance < self.test_ratio * suboptimal_value.distance:
                good_matches.append(optimal_value)

        return good_matches


    @staticmethod
    def PerspectiveTransformImage(img_right, key_points_left, key_points_right, good_matches):
        # Load feature points
        left_points = np.float32([key_points_left[m.queryIdx].pt for m in good_matches])
        left_points = left_points.reshape(-1, 1, 2)
        right_points = np.float32([key_points_right[m.trainIdx].pt for m in good_matches])
        right_points = right_points.reshape(-1, 1, 2)

        # Calculate homography matrix for coordinate transformation
        homography_matrix, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, 2)
        
        # Perspective transformation
        rigth_img_warp = cv2.warpPerspective(img_right, np.linalg.inv(homography_matrix),
                                             (img_right.shape[1] + img_right.shape[1],
                                              img_right.shape[0])
                                              )
        
        return rigth_img_warp


    @staticmethod
    def FindOverlapAreaBorder(cols, img_left, rigth_img_warp):
        left_border = 0
        right_border = cols
        
        # Left border
        for col in range(0, cols):
            if img_left[:, col].any() and rigth_img_warp[:, col].any():
                left_border = col
                break
        # Right border
        for col in range(cols - 1, 0, -1):
            if img_left[:, col].any() and rigth_img_warp[:, col].any():
                right_border = col
                break

        return left_border, right_border
    

    @staticmethod
    def FuseImage(img_left, rigth_img_warp, rows, cols, left_border, right_border):
        img_fusion = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                # Non-overlapping area are assigned directly
                if not img_left[row, col].any():
                    img_fusion[row, col] = rigth_img_warp[row, col]
                elif not rigth_img_warp[row, col].any():
                    img_fusion[row, col] = img_left[row, col]
                    
                # Weighted average of overlapping area
                else:
                    # Calculate weight
                    distance_left_border = float(abs(col - left_border))
                    distance_right_border = float(abs(col - right_border))
                    alpha = distance_left_border / (distance_left_border + distance_right_border)

                    # Fusion
                    pixel_new = img_left[row, col] * (1 - alpha) + \
                                rigth_img_warp[row, col] * alpha
                    img_fusion[row, col] = np.clip(pixel_new, 0, 255)
        
        return img_fusion


    def SpliceImage(self, img_left, img_right, key_points_left, key_points_right, good_matches):
        # Perspective transform the right image to align the splice
        rigth_img_warp = self.PerspectiveTransformImage(img_right, key_points_left,
                                                        key_points_right, good_matches
                                                        )

        # Get left image shape
        rows, cols = img_left.shape[:2]
 
        # Find the boundary of the image overlap area
        left_border, right_border = self.FindOverlapAreaBorder(cols, img_left, rigth_img_warp)

        # Fuse image
        img_fusion = self.FuseImage(img_left, rigth_img_warp,
                                    rows, cols, left_border, right_border
                                    )
        rigth_img_warp[0:img_left.shape[0], 0:img_left.shape[1]] = img_fusion

        return rigth_img_warp


    @staticmethod
    def RemoveBlackColumn(img_fusion):
        for i in range(img_fusion.shape[1] - 1, -1, -1):
            if np.all(np.all(img_fusion[:, i] == [0, 0, 0], axis = 1)):
                img_fusion = np.delete(img_fusion, i, 1)
            else:
                break
        return img_fusion


    def Run(self):
        if len(self.imgs) >= 2:
            img_fusion = None
            img_right = cv2.imread(self.img_path + self.imgs[0])
            for i in range(1, len(self.imgs)):
                # Load image
                img_left = cv2.imread(self.img_path + self.imgs[i])

                # Extract feature points
                key_points_left, describe_left = self.sift.detectAndCompute(img_left, None)
                key_points_right, describe_right = self.sift.detectAndCompute(img_right, None)

                # Match feature points
                matches = self.MatchFeaturePoints(describe_left, describe_right)

                # Check match result
                good_matches = self.CheckMatches(matches)

                # Show match result
                if self.show_matches:
                    img_matches = cv2.drawMatches(img1 = img_left, keypoints1 = key_points_left,
                                                  img2 = img_right, keypoints2 = key_points_right,
                                                  matches1to2 = good_matches, outImg = None
                                                  )
                    cv2.imshow("matches", img_matches)
                    cv2.waitKey(1000)

                # Splice image
                if len(good_matches) > self.match_points_min:
                    img_fusion = self.SpliceImage(img_left, img_right,
                                                  key_points_left, key_points_right,
                                                  good_matches
                                                  )
                else:
                    print("[bold red][ERROR] Lack of matching points![/bold red]")
                    break

                # Update left image
                img_right = img_fusion

            # Save image
            if img_fusion is not None:
                img_fusion = self.RemoveBlackColumn(img_fusion)

                cv2.imwrite(self.img_name, img_fusion)
                print(f"[bold green][INFO] The mosaic image {self.img_name} has been saved.[/bold green]")
        else:
            print("[bold red][ERROR] Lack of matching iamges![/bold red]")


if __name__=="__main__":
    image_splicer = ImageSplicer()
    image_splicer.Run()
