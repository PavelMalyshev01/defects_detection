def upload_images(
        start: bool,
        train_path: str,
        val_path: str
):
    """
    :code_assign: users
    :code_type: Пользовательские функции
    :imports: init_gui_dict
    :packages:
    import os
    :param_block bool start: Запуск
    :param str train_path: Директория с обучающими изображениями и метками
    :param str val_path: Директория с валидационными изображениями и метками
    :returns: train_path_ret, val_path_ret, error
    :rtype: str, str, str
    :semrtype: , ,
    """

    error = ''
    train_path_ret = train_path
    val_path_ret = val_path

    if not os.path.exists(train_path_ret):
        error = 'Директория с обучающими изображениями не найдена'
        return None, None, error
    if not os.path.exists(val_path_ret):
        error = 'Директория с валидационными изображениями не найдена'
        return None, None, error

    return train_path_ret, val_path_ret, error
