def cam_calibration(
    start: bool,
    images_path: str,
    board_width: int,
    board_height: int
):
    """
    :code_assign: users
    :code_type: Пользовательские функции
    :imports: init_gui_dict, image_to_base64
    :packages:
    import numpy as np
    import cv2
    import glob
    :param_block bool start: Запуск
    :param str images_path: Директория с изображениями шахматной доски
    :param int board_width: Количество углов по ширине
    :param int board_height: Количество углов по высоте
    :returns: calibration_file_path, gui_dict, error
    :rtype: str, dict, str
    :semrtype: , ,
    """
    error = ''
    gui_dict = init_gui_dict()

    # Создание сетки координат шахматной доски
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

    # Массивы для хранения точек шахматной доски и 3D точек
    objpoints = []  # 3D точки в реальном мире
    imgpoints = []  # 2D точки на изображении

    # Путь к изображениям шахматной доски
    images = glob.glob(images_path + '/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Поиск углов шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)

        # Если углы найдены, добавляем их в массив
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Отрисовка углов на изображении
            cv2.drawChessboardCorners(img, (board_width, board_height), corners, ret)
            cv2.imwrite(images_path + '/corners.png', img)

    try:
        # сохраняем изображение
        img_corners = image_to_base64(images_path + '/corners.png', "png")
    # ошибка
    except Exception as err:
        error = f'Ошибка отрисовки углов на изображении: {err}'
        return None, None, error

    # записываем изображение с отрисованными углами в gui_dict
    gui_dict['image'].append({'title': 'Отрисовка углов', 'value': img_corners})

    # Калибровка камеры
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    gui_dict['text'].append({'title': 'Параметры калибровки',
                             'value': 'Матрица камеры:\n' + str(mtx) + ',\n' +
                                      'Коэффициенты дисторсии:\n' + str(dist) + ',\n' +
                                      'Векторы вращения:\n' + str(rvecs) + ',\n' +
                                      'Векторы трансляции:\n' + str(tvecs)})

    # Сохранение параметров калибровки в файл
    calibration_file_path = images_path + '/calibration.npz'
    np.savez(calibration_file_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, objpoints=objpoints, imgpoints=imgpoints)

    return calibration_file_path, gui_dict, error
