def enchance_contrast(
        folder_path: str,
        make_clahe: bool
):
    """
    :code_assign: users
    :code_type: Пользовательские функции
    :imports: init_gui_dict
    :packages:
    import os
    import cv2
    :param_block str folder_path: Директория с обучающими изображениями и метками
    :param bool make_clahe: Нужно ли выполнять CLAHE
    :returns: save_path, gui_dict, error
    :rtype: str, dict, str
    :semrtype: , ,
    """

    error=''
    gui_dict = init_gui_dict()

    if make_clahe:
        images_folder = folder_path + '/images'
        labels_folder = folder_path + '/labels'

        # Удаление старых результатов повышения контрастности
        for filename in os.listdir(images_folder):
            if filename.endswith("_contrast.jpg") or filename.endswith("_contrast.png"):
                os.remove(os.path.join(images_folder, filename))
        for filename in os.listdir(labels_folder):
            if filename.endswith("_contrast.txt"):
                os.remove(os.path.join(labels_folder, filename))

        # Обработка каждого изображения в папке "images"
        for filename in os.listdir(images_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Форматы изображений, которые вы хотите обработать
                input_path = os.path.join(images_folder, filename)

                # Загрузка цветного изображения
                image = cv2.imread(input_path)

                # Проверка на успешную загрузку изображения
                if image is None:
                    print(f"Ошибка загрузки изображения: {input_path}")
                    continue

                # Преобразование изображения в формат YUV (для применения CLAHE ко всем каналам цвета)
                image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

                # Применение CLAHE к каналу яркости (Y)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])

                # Преобразование изображения обратно в формат BGR
                enhanced_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

                # Получение имени файла без расширения
                filename_without_extension, extension = os.path.splitext(filename)

                # Формирование нового имени файла для улучшенного изображения
                output_filename = f"{filename_without_extension}_contrast{extension}"

                # Полный путь для сохранения улучшенного изображения
                output_path = os.path.join(images_folder, output_filename)

                # Сохранение улучшенного изображения в той же папке с новым именем файла
                cv2.imwrite(output_path, enhanced_image)

        # Обработка каждого файла в папке "labels"
        for filename in os.listdir(labels_folder):
            if filename.endswith(".txt"):  # Формат файлов, которые вы хотите обработать
                input_path = os.path.join(labels_folder, filename)

                # Чтение содержимого исходного файла меток
                with open(input_path, 'r') as f:
                    content = f.read()

                # Получение имени файла без расширения
                filename_without_extension, extension = os.path.splitext(filename)

                # Формирование нового имени файла для копии с суффиксом "_contrast"
                output_filename = f"{filename_without_extension}_contrast{extension}"

                # Полный путь для сохранения копии файла с суффиксом "_contrast"
                output_path = os.path.join(labels_folder, output_filename)

                # Запись содержимого копии файла с суффиксом "_contrast"
                with open(output_path, 'w') as f:
                    f.write(content)

    save_path = folder_path

    return save_path, gui_dict, error
