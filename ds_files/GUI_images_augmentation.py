{
    'images_augmentation':
        {
            'hint':
                {
                    'rus': 'Аугментация изображений',
                },
            'blocks': 'Process',
            'gui_name':
                {
                    'rus': 'Аугментация изображений',
                },
            'in_params':
                {
                    'images_path':
                        {
                            'gui_name':
                                {
                                    'rus': 'Датасет с изображениями'
                                },
                            'gui_type': 'api_fs_folder',
                        },
                    'make_aug':
                        {
                            'gui_name':
                                {
                                    'rus': 'Выполнить аугментацию'
                                },
                            'gui_type': 'checkbox'
                        },
                    'aug_n':
                        {
                            'gui_name':
                                {
                                    'rus': 'Количество преобразований'
                                },
                            'gui_type': 'input',
                            'gui_default_values':
                                {
                                    'rus': 10
                                },
                            'gui_range': [0, 100, 1],
                            'gui_visible':
                                {
                                    'make_aug':
                                        {
                                            1: True,
                                        },
                                },
                        },
                },
            'out_params':
                {
                    'save_path':
                        {
                            'gui_name':
                                {
                                    'rus': 'Обработанный датасет'
                                }
                        },
                }
        }
}
