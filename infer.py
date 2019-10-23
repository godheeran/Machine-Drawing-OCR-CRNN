import numpy as np
import os
import shutil


def split_infer(X, train_size):
    # split validation data
    total_size = len(X)

    infer_indices = np.random.choice(total_size, train_size, replace=False)
    X_infer = X[infer_indices]

    return X_infer

def load_img_path(images_path):
    tmp = os.listdir(images_path)
    tmp.sort(key=lambda x: int(x.split('.')[0]))

    file_names = [images_path + s for s in tmp]

    file_names = np.asarray(file_names)

    return file_names

def cp_file(imgs_list_para, dst_para):
    for i in range(imgs_list_para.shape[0]):
        file_path = imgs_list_para[i]

        filename = os.path.basename(file_path)
        fn = filename.split('.')[0]
        ext = filename.split('.')[1]

        dest_filename = dst_para + fn + '.' + ext

        shutil.copyfile(file_path, dest_filename)


if __name__ == '__main__':
    images_path = './imgs/ResizedImg28X28/'
    image_path_list = load_img_path(images_path)
    # print(image_path_list[:10])

    X_infer = split_infer(image_path_list, 10000)
    
    cp_file(X_infer, './imgs/infer/')
    
