class Path(object):
    @staticmethod
    def db_dir(database):
        # print('111111111')
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'E:\\eclipse-workspace\\PyTorch\\pytorch-video-recognition-master\\data\\UCF-101'

            # Save preprocess data into output_dir
            output_dir = 'E:\\eclipse-workspace\\PyTorch\\pytorch-video-recognition-master\\data_process\\ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        
        elif database == 'Auslan':

            root_dir = 'Data_ori/Auslan_Dataset'
            output_dir = 'Data_processed'
            
            return root_dir, output_dir


        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './model/c3d-pretrained.pth'