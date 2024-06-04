def images_calibration(
    calibration_file_path: str,
    images_path: str
):
    """
    :code_assign: users
    :code_type: Пользовательские функции
    :imports: init_gui_dict
    :packages:
    import numpy as np
    import cv2
    import os
    :param_block str calibration_file_path: Путь к файлу с параметрами калибровки
    :param_block str images_path: Директория с изображениями
    :returns: images_path, gui_dict, error
    :rtype: str, dict, str
    :semrtype: , ,
    """
    error = ''
    gui_dict = init_gui_dict()

    # Загрузка параметров калибровки из файла
    calibration_data = np.load(calibration_file_path)
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']

    for filename in os.listdir(images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Полный путь к файлу изображения
            img_path = os.path.join(images_path, filename)
            # Загрузка исходного изображения
            img = cv2.imread(img_path)
            # Исправление искажений
            undistorted_img = cv2.undistort(img, mtx, dist, None)

    # Загрузка массивов objpoints и imgpoints
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
    objpoints = calibration_data['objpoints']
    imgpoints = calibration_data['imgpoints']
    rvecs = calibration_data['rvecs']
    tvecs = calibration_data['tvecs']

    # Вычисление Re-projection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints_reproj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        re_error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
        mean_error += re_error

    mean_error /= len(objpoints) * 10

    gui_dict['text'].append({'title': 'Ошибка проецирования',
                             'value': 'Re-projection Error = ' + str(mean_error)})


    return images_path, gui_dict, error
