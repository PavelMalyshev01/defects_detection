def images_augmentation(
        images_path: str,
        make_aug: bool=True,
        aug_n: int=10
):
    """
    :code_assign: users
    :code_type: Пользовательские функции
    :imports: init_gui_dict
    :packages:
    import os
    import cv2
    import shutil
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    :param_block str images_path: Директория с обучающими изображениями и метками
    :param bool make_aug: Нужно ли выполнять аугментацию
    :param int aug_n: Количество преобразований
    :returns: save_path, gui_dict, error
    :rtype: str, dict, str
    :semrtype: , ,
    """

    error = ''
    gui_dict = init_gui_dict()

    save_path = images_path + '_aug'

    if make_aug:
        # Удаление папки с обработанными изображениями
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        # Путь к папкам с изображениями и ограничивающими рамками
        images_folder = images_path + '/images'
        labels_folder = images_path + '/labels'
        output_images_folder = save_path + '/images'
        output_labels_folder = save_path + '/labels'

        # Создание папок для аугментированных изображений и ограничивающих рамок
        os.makedirs(output_images_folder)
        os.makedirs(output_labels_folder)

        # Копирование исходных изображений и меток в папки для аугментированных данных
        shutil.copytree(images_folder, os.path.join(output_images_folder), dirs_exist_ok=True)
        shutil.copytree(labels_folder, os.path.join(output_labels_folder), dirs_exist_ok=True)

        # Загрузка изображений и их соответствующих меток
        image_files = os.listdir(images_folder)
        for image_file in image_files:
            # Загрузка изображения
            image_path = os.path.join(images_folder, image_file)
            image = cv2.imread(image_path)

            # Загрузка меток
            label_file = os.path.join(labels_folder, os.path.splitext(image_file)[0] + ".txt")
            with open(label_file, "r") as f:
                lines = f.readlines()
                boxes = []
                for line in lines:
                    class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split(" "))
                    # Преобразование координат в формат, подходящий для imgaug
                    x1 = int((x_center - box_width / 2) * image.shape[1])
                    y1 = int((y_center - box_height / 2) * image.shape[0])
                    x2 = int((x_center + box_width / 2) * image.shape[1])
                    y2 = int((y_center + box_height / 2) * image.shape[0])
                    boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

            for i in range(aug_n):
                seq = iaa.Sequential([
                    iaa.Fliplr(0.5),  # Отражение по горизонтали
                    iaa.Affine(rotate=(-10, 10)),  # Поворот на случайный угол от -10 до 10 градусов
                    iaa.Affine(scale=(0.8, 1.2)),  # Масштабирование изображения на случайный коэффициент от 0.8 до 1.2
                    iaa.GaussianBlur(sigma=(0, 3.0)),
                    # Применить гауссовское размытие со случайным значением сигмы от 0 до 3.0
                    iaa.Dropout(p=(0, 0.002)),  # Выпадение случайного количества пикселей с вероятностью от 0 до 0.002
                    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
                    # Добавить гауссовский шум со случайной амплитудой от 0 до 0.05*255
                    iaa.Add((-10, 10)),  # Добавить случайное значение от -10 до 10 ко всем пикселям
                    iaa.Multiply((0.9, 1.1)),  # Умножить значения пикселей на случайный коэффициент от 0.5 до 1.5
                    iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25),
                    # Применить эластичное искажение со случайными параметрами
                    iaa.PerspectiveTransform(scale=(0.01, 0.1))
                    # Применить перспективное искажение со случайным масштабированием от 0.01 до 0.1
                ])
                augmented_image, augmented_boxes = seq(image=image, bounding_boxes=boxes)

                # Сохранение аугментированного изображения
                output_image_path = os.path.join(output_images_folder, "aug_" + image_file.split('.')[0] + f"_{i}.jpg")
                cv2.imwrite(output_image_path, augmented_image)

                # Сохранение аугментированных меток
                output_label_file = os.path.join(output_labels_folder,
                                                 'aug_' + os.path.splitext(image_file)[0] + f'_{i}.txt')
                with open(output_label_file, "w") as f:
                    for box in augmented_boxes:
                        # Преобразование координат обратно в исходный формат
                        x_center = (box.x1 + box.x2) / (2 * augmented_image.shape[1])
                        y_center = (box.y1 + box.y2) / (2 * augmented_image.shape[0])
                        box_width = (box.x2 - box.x1) / augmented_image.shape[1]
                        box_height = (box.y2 - box.y1) / augmented_image.shape[0]
                        f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")


    return save_path, gui_dict, error
