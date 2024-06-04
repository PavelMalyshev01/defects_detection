{
    'enchance_contrast':
        {
            'hint':
                {
                    'rus': 'Повышение контрастности CLAHE',
                },
            'blocks': 'Process',
            'gui_name':
                {
                    'rus': 'CLAHE',
                },
            'in_params':
                {
                    'folder_path':
                        {
                            'gui_name':
                                {
                                    'rus': 'Датасет с изображениями'
                                },
                            'gui_type': 'api_fs_folder',
                        },
                    'make_clahe':
                        {
                            'gui_name':
                                {
                                    'rus': 'Выполнить повышение контрастности'
                                },
                            'gui_type': 'checkbox'
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
