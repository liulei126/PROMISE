import os
from skimage.exposure import equalize_adapthist
# from metrics import dice_coef, dice_coef_loss
from augmenters import *

train_mhd_path = '/mnt/data3/llei/PROMISE12/TrainingData/'
train_mhd_aug_path = '/mnt/data3/llei/PROMISE12/TrainingData_aug/'
test_mhd_path = '/mnt/data3/llei/PROMISE12/TestData/'


def img_resize(imgs, img_rows, img_cols, equalize=True):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )
            # img = clahe.apply(cv2.convertScaleAbs(img))
        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs


def divide_data_to_array(img_rows, img_cols):
    # train data
    fileList = os.listdir(train_mhd_path)
    fileList_aug = os.listdir(train_mhd_aug_path)
    fileList += fileList_aug
    fileList = list(filter(lambda x: '.mhd' in x, fileList))
    fileList.sort()
    # mean: 0.4128385365774754 std: 0.2467691380265403
    mu = 0.4128385365774754
    sigma = 0.2467691380265403
    if not os.path.exists('train_data'):
        os.makedirs('train_data')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    for filename in fileList:
        print(filename)
        if 'aug' in filename:
            itkimage = sitk.ReadImage(train_mhd_aug_path + filename)
        else:
            itkimage = sitk.ReadImage(train_mhd_path + filename)
        imgs = sitk.GetArrayFromImage(itkimage)
        # print(imgs.shape)(47, 512, 512)
        if 'segm' in filename.lower():
            imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
            # print(imgs.shape)(47, 512, 512)
            imgs = imgs.reshape(-1, img_rows, img_cols, 1)
            # print(imgs.shape)(47, 512, 512, 1)
            masks = imgs.astype(int)
            np.save('train_data/'+filename.split('.')[0]+'.npy', masks)
        else:
            imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
            # print(imgs.shape)(47, 512, 512)
            imgs = imgs.reshape(-1, img_rows, img_cols, 1)
            # print(imgs.shape)(47, 512, 512, 1)
            # Smooth images using CurvatureFlow
            images = smooth_images(imgs)
            # mean: 0.4128385365774754 std: 0.2467691380265403
            images = (images - mu) / sigma
            np.save('train_data/'+filename.split('.')[0]+'.npy', images)
        # break

    # test data
    fileList =  os.listdir(test_mhd_path)
    fileList = list(filter(lambda x: '.mhd' in x, fileList))
    fileList.sort()
    for filename in fileList:
        print(filename)
        itkimage = sitk.ReadImage(test_mhd_path + filename)
        imgs = sitk.GetArrayFromImage(itkimage)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=True).reshape(-1, img_rows, img_cols, 1)
        images = smooth_images(imgs)
        images = (images - mu)/sigma
        np.save('test_data/'+filename.split('.')[0]+'.npy', images)
        # break

if __name__=='__main__':

    import time

    start = time.time()
    divide_data_to_array(256, 256)

    end = time.time()

    print('Elapsed time:', round((end-start)/60, 2))