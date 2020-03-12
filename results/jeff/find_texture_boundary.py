import time
import cv2 
import numpy as np 
  
img = cv2.imread('../me.png', cv2.IMREAD_COLOR)  
print("img  :   ", img.shape, img.shape[0], img[0][0])

update_img = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
cv2.circle(img, (0,0), 5,(255,255,255))
cv2.circle(img, (0,img.shape[1]), 5,(0,0,255))
cv2.circle(img, (img.shape[0],0), 5,(0,0,255))
cv2.circle(img, (img.shape[0], img.shape[1]), 5,(0,0,0))

boundary_xy = []


def get_min_max(values):
    return [values.index(min(values)), values.index(max(values))]


cv2.imshow("Detected Boundary", img)

for y in range(img.shape[1]):  # img.shape[0]
    d_arr = []
    xy_arr = []
    for x in range(img.shape[0]):
        if np.count_nonzero(img[x][y]) > 0: 
        # if img[i][j].tolist() != [167, 144, 213]:
            d_arr.append(np.linalg.norm(np.array([0,y])-np.array([x, y])))
        else:
            d_arr.append(-1)
        # print("-> img[%d][%d]" %(x, y))
        update_img[y, x] = img[y, x]
        # cv2.imshow("Updated Boundary", update_img)
        # cv2.waitKey(0)
    if max(d_arr) < 0:
        continue
    min_max = get_min_max(d_arr)
    xy_arr = [[min_max[0], y], [min_max[1], y]]
    boundary_xy += xy_arr
    
    print("## img[%d][%d]: %d %s" %(x, y, len(xy_arr), xy_arr))
    cv2.circle(update_img, tuple(xy_arr[0]), 5,(0,0,255))
    cv2.circle(update_img, tuple(xy_arr[1]), 5,(255,0,0))
    cv2.imshow("Updated Boundary", update_img)
    cv2.waitKey(0)

    # boundary_xy.append([i, min_max[0]])
    # boundary_xy.append([i, min_max[1]])
    
    # cv2.putText(img, '#', tuple([i, min_max[1]]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 100,(0,0,255))
    # cv2.circle(img, tuple([i, min_max[0]]), 1,(0,0,255))
    # time.sleep(6) 
    
cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(boundary_xy))
print(boundary_xy[0], boundary_xy[len(boundary_xy) - 1])


# for point in boundary_xy:
#     cv2.circle(img, tuple(point), 1,(0,0,255))

# cv2.imshow("Detected Boundary 2", img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
