{
    'cam_calibration':
        {
            'hint':
                {
                    'rus': 'Определение параметров калибровки',
                },
            'blocks': 'Process',
            'gui_name':
                {
                    'rus': 'Определение параметров калибровки',
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
                    'images_path':
                        {
                            'gui_name':
                                {
                                    'rus': 'Директория с изображениями шахматной доски'
                                },
                            'gui_type': 'api_fs_folder',
                        },
                    'board_width':
                        {
                            'gui_name':
                                {
                                    'rus': 'Количество углов по ширине'
                                },
                            'gui_type': 'input',
                            'gui_type_value': 'number',
                            'gui_default_values':
                                {
                                    'rus': 7
                                },
                            'gui_range': [1, 'Infinitive', 1],
                        },
                    'board_height':
                        {
                            'gui_name':
                                {
                                    'rus': 'Количество углов по высоте'
                                },
                            'gui_type': 'input',
                            'gui_type_value': 'number',
                            'gui_default_values':
                                {
                                    'rus': 7
                                },
                            'gui_range': [1, 'Infinitive', 1],
                        },
                },
            'out_params':
                {
                    'calibration_file_path':
                        {
                            'gui_name':
                                {
                                    'rus': 'Параметры калибровки'
                                }
                        },
                }
        }
}
