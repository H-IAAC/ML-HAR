import torchvision.transforms as transforms
import datasets.omniglot as om
import os
import sys

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all=False):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if path is None:
                return om.Omniglot("../data/omni", background=background, download=True, train=train,
                                   transform=train_transform, all=all)
            else:
                return om.Omniglot(path, background=background, download=True, train=train,
                                   transform=train_transform, all=all)


        else:
            print("Unsupported Dataset")
            assert False

    @staticmethod
    def get_dataset_conf(dataset):
        # data_points = int(time_window * freq)
        # data_size = (sensors,int(.data_points))
        path = os.path.dirname(os.getcwd()) + '/datasets/' + dataset + '/'
        print('PATH ', path)
        if dataset == "hapt":

            return [
                {"name": "hapt","hapt": "HumanActivityRecognition", "sensors":  6, "time_window": 2.56, "freq":50, "data_points": int(2.56 * 50),"data_size": (50, int(2.56*50)),
                "labels_id": {1:'walking',2:'walking upstairs',3:'walking downstairs',4: 'sitting',5: 'standing',6: 'laying',7: 'stand_to_sit',8: 'sit_to_stand',9: 'sit_to_lie',10: 'lie_to_sit', 11: 'stand_to_lie',
                 12:'lie_to_stand'}, "path": path } 
               ]

        elif dataset == "ucihar":
             return [
                 {"name": "ucihar","ucihar": "HumanActivityRecognition", "sensors":  9, "time_window": 2.56, "freq":50, "data_points": int(2.56 * 50),"data_size": (50, int(2.56*50)),
                "labels_id": {1:'walking',2:'walking upstairs',3:'walking downstairs',4: 'sitting',5: 'standing',6: 'laying'}, 
                "path": path} 
             ]
    
        elif dataset == "dsads":
             return [
                {"name": "dsads", "dsads": "HumanActivityRecognition", "sensors":  45, "time_window": 5, "freq":25, "data_points": int(5 * 25),"data_size": (25, int(5*50)),
                "labels_id": {1: 'sitting', 2: 'standing',3: 'lying on back', 4: 'lying on right side', 5: 'ascending stairs',
                                  6: 'descending stairs', 7: 'standing in an elevator still', 8:'moving around in an elevator', 
                                  9: 'walking in a parking lot', 10: 'walking on a treadmill with a speed of 4 km/h (in flat)', 
                                  11: 'walking on a treadmill with a speed of 4 km/h (in 15 deg inclined positions)', 
                                  12: 'running on a treadmill with a speed of 8 km/h' , 13: 'exercising on a stepper', 
                                  14: 'exercising on a cross trainer', 15:'cycling on an exercise bike in horizontal position', 
                                  16: 'cycling on an exercise bike in vertical positions', 17:'rowing' , 18: 'jumping', 
                                  19: 'playing basketball'}, 
                "path": path} 
               ]            
 
        elif dataset == "pamap2":
             return [
                {"name": "pamap2", "pamap2": "HumanActivityRecognition", "sensors":  31, "time_window": 5.2, "freq":20, "data_points": int(5.2 * 20),"data_size": (31, int(5.2*20)),
                "labels_id": {1:'lying',2:'sitting',3:'standing',4: 'walking',5: 'running',6: 'cycling',7: 'Nordic walking',
                             9: 'watching TV', 10: 'computer work',11: 'car driving', 12: 'ascending stairs', 13:'descending stairs',
                             16: 'vacuum cleaning',17: 'ironing',18: 'folding laundry',19: 'house cleaning',20:'playing soccer',
                             24: 'rope jumping'}, 
                "path": path } 
               ]      
 
        else:
            print("Unsupported dataset; either implement the dataset or choose a different dataset")
            assert (False)
            sys.exit()