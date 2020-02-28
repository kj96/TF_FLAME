import time
import cv2 
import numpy as np 
  
img = cv2.imread('jeff1_face.png', cv2.IMREAD_COLOR)  
print("img  :   ", img.shape, img.shape[0], img[0][0])

boundary_xy = []


def get_min_max(values):
    return [values.index(min(values)), values.index(max(values))]


cv2.imshow("Detected BOundary", img)

for i in range(img.shape[0]):  # img.shape[0]
    d_arr = []
    for j in range(img.shape[1]):
        if np.count_nonzero(img[i][j]) > 0:
            # print("img[i][j]: ", img[i][j])
            d_arr.append(np.linalg.norm(np.array([i,0])-np.array([i,j])))
        else:
            d_arr.append(-1)
    if max(d_arr) < 0:
        continue
    min_max = get_min_max(d_arr)
    boundary_xy.append([i, min_max[0]])
    boundary_xy.append([i, min_max[1]])
    
    cv2.putText(img, '#', tuple([i, min_max[1]]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 100,(0,0,255))
    cv2.circle(img, tuple([i, min_max[0]]), 1,(0,0,255))
    # time.sleep(6) 
    
cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(boundary_xy))
print(boundary_xy[0], boundary_xy[len(boundary_xy) - 1])


# for point in boundary_xy:
#     cv2.circle(img, tuple(point), 1,(0,0,255))

# cv2.imshow("Detected BOundary", img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
