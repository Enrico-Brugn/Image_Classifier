


def cut(A):
    min_pl = min_pool(A)[A.shape[0]//3:2*A.shape[0]//3]
    # plt.imshow(min_pl, cmap='Greys_r')
    # plt.show()
    max_pl = max_pool(A)[A.shape[0]//3:2*A.shape[0]//3]
    # plt.imshow(min_pl, cmap='Greys_r')
    # plt.show()
    
    avg_min = np.min(min_pl, axis=0)
    avg_max = np.max(max_pl, axis=0)
    # plt.plot(avg_min)
    # plt.plot(avg_max)
    # plt.show()
    
    grad_min = np.abs(np.gradient(avg_min))
    grad_max = np.abs(np.gradient(avg_max))
    # plt.plot(grad_min)
    # plt.plot(grad_max)
    # plt.show()

    cut_point_1 = np.argmax(grad_min)+1
    cut_point_2 = np.argmax(grad_max)+1
    
    print(cut_point_1,cut_point_2)
    cut_point = cut_point_1 if cut_point_1>cut_point_2 else cut_point_2

    img1 = A[:,cut_point :]
    return img1

def cut_4(img1):
    
    img1 = cut(img1)
    img1 = np.fliplr(img1)
    img1 = cut(img1)
    
    
    
    
    img1 = np.rot90(img1)
    img1 = cut(img1)
    return img1
    
    img1 = np.fliplr(img1)
    img1 = cut(img1)
    img1 = np.fliplr(img1)
    img1 = np.rot90(img1,3)
   
    img2 = np.fliplr(img1)
    
    return img1, img2

def split(img1):
    arrays = np.array_split(img1,3,axis=1)
    return arrays


work_img = copy.deepcopy(img)
arr1 = cut_4(work_img)
plt.imshow(arr1, cmap='Greys_r')
plt.show()

# arr2 = split(arr1)
# plt.imshow(arr2[0], cmap='Greys_r')
# plt.show()
# plt.imshow(arr2[1], cmap='Greys_r')
# plt.show()
# plt.imshow(arr2[2], cmap='Greys_r')
# plt.show()
