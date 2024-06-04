{
    'custom_yolov5_train':
        {
            'hint':
                {
                    'rus': 'Обнаружение и классификация объектов. Обучение YOLOv5 на своих данных.',
                },
            'blocks': 'Process',
            'gui_name':
                {
                    'rus': 'Custom YOLOv5s',
                },
            'in_params':
                {
                    'images_path_train':
                        {
                            'gui_name':
                                {
                                    'rus': 'Датасет для обучения',
                                },
                        },
                    'images_path_val':
                        {
                            'gui_name':
                                {
                                    'rus': 'Датасет для валидации',
                                },
                        },
                    'batch_size':
                        {
                            'gui_name':
                                {
                                    'rus': 'Размер мини-батча',
                                },
                            'gui_type': 'input',
                            'gui_type_value': 'number',
                            'gui_default_values':
                                {
                                    'rus': 1
                                },
                            'gui_range': [1, 'Infinitive', 1],
                        },
                    'epochs':
                        {
                            'gui_name':
                                {
                                    'rus': 'Количество эпох',
                                },
                            'gui_type': 'input',
                            'gui_type_value': 'number',
                            'gui_default_values':
                                {
                                    'rus': 5
                                },
                            'gui_range': [1, 'Infinitive', 1],
                        },
                },
            'out_params':
                {
                    'weights':
                        {
                            'gui_name':
                                {
                                    'rus': 'Модель YOLOv5'
                                },
                        },
                }

        }
}