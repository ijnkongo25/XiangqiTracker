class Path(object):
    @staticmethod
    def db_dir(database):
        
        if database == 'Chinese_chess':

            root_dir = 'E:\AMME4710_major\CNN\Data_ori_all\Chinese_chess_database'
            # root_dir = 'E:\AMME4710_major\CNN\Data_ori_traffic\Chinese_chess_database'
            # root_dir = 'E:\AMME4710_major\CNN\Data_ori\Chinese_chess_database'

            output_dir = 'E:\AMME4710_major\CNN\Data_processed'
            
            return root_dir, output_dir


        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './model/c3d-pretrained.pth'