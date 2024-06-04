{
    'upload_images':
        {
            'hint':
                {
                    'rus': 'Загрузка изображений для OD',
                },
            'blocks': 'Process',
            'gui_name':
                {
                    'rus': 'Загрузка изображений для OD',
                },
            'in_params':
                {
                    'start':
                        {
                            'gui_name':
                                {
                                    'rus': 'Запуск'
                                }
                        },
                    'train_path':
                        {
                            'gui_name':
                                {
                                    'rus': 'Директория с обучающими изображениями и метками'
                                },
                            'gui_type': 'api_fs_folder',
                        },
                    'val_path':
                        {
                            'gui_name':
                                {
                                    'rus': 'Директория с валидационными изображениями и метками'
                                },
                            'gui_type': 'api_fs_folder',
                        },
                },
            'out_params':
                {
                    'train_path_ret':
                        {
                            'gui_name':
                                {
                                    'rus': 'Датасет для обучения'
                                }
                        },
                    'val_path_ret':
                        {
                            'gui_name':
                                {
                                    'rus': 'Датасет для валидации'
                                },
                        },
                }
        }
}
