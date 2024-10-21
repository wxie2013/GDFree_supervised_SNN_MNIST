import os, torch, torchvision, ray
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from filelock import FileLock

raw_amp_scale = 255.0  # torchvision normalized pixel amp to 0-1. This is to recover the raw amp
class fetch_torchvision_data():
    def __init__(self, 
                 idx_start_train, idx_end_train, 
                 idx_start_valid, idx_end_valid,
                 idx_start_test, idx_end_test, 
                 which_data, frac_val = 0.2, 
                 fixed_seed = True, num_workers = 1):
        '''
            idx_start_train, idx_end_train: start and end index of training set
            idx_start_valid, idx_end_valid: start and end index of validation set
            idx_start_test, idx_end_test: start and end index of test set
            which_data: name of the dataset, e.g. MNIST
            frac_val: fraction of validation set
            fixed_seed: whether to fix the random seed
        '''

        if fixed_seed == True: # for reproducibility
            torch.manual_seed(0)

        self.num_workers = num_workers
        self.frac_val = frac_val
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.which_data = which_data
        self.idx_start_train, self.idx_end_train = idx_start_train, idx_end_train
        self.idx_start_valid, self.idx_end_valid = idx_start_valid, idx_end_valid
        self.idx_start_test, self.idx_end_test = idx_start_test, idx_end_test

        # in itap cluster
        self.data_dir = "/depot/wxie/data"
        # in local PC
        if os.path.exists(self.data_dir) == False:
            self.data_dir = "/home/wxie/Brian2"

        self.data_dir = os.path.join(self.data_dir, "torchvision_data", self.which_data)
        if os.path.exists(self.data_dir) == False:
            os.makedirs(self.data_dir)

        # compile data
        self.compile_data()

    # check if the sample is an integer number of num_workers.
    # the number of samples need to be an integer-fold of num_workers in my model,
    # e.g. 0-10 for 3 workers will  create 4 samples, with first 3 content 3 elements
    # and last one content 1 element. Because my model will require reading from raw
    # when idx_start = 0, it will create two separate seg_dir, i.e. seg_0_3 and seg_0_1
    # both start from raw.
    def check_range(self, idx_start, idx_end):
        new_idx_end = idx_start + self.num_workers*((idx_end-idx_start)//self.num_workers)
        if new_idx_end != idx_end:
            print(f'--- range changed:  idx_start: {idx_start},  idx_end: {new_idx_end} ---')
        
        return idx_start, new_idx_end


    # compile trainset, valset, testset  into a dictionary of numpy arrays
    def compile_data(self):
        with FileLock(f"{self.data_dir}.lock"):
            if self.which_data == "MNIST":
                torchvision_data = datasets.MNIST
            elif self.which_data == "FashionMNIST":
                torchvision_data = datasets.FashionMNIST
            elif self.which_data == "CIFAR10":
                torchvision_data = datasets.CIFAR10
            else:
                sys.exit('!!! The data for', self.which_data, ' is not defined yet. Please define it first here!!!')

            # Split data into train and val sets
            data_train_val = torchvision_data(root=self.data_dir, train=True, download=True, transform=self.transform)
            trainset, valset = random_split(data_train_val, [1-self.frac_val, self.frac_val])

            # train data
            if self.idx_end_train > len(trainset):
                self.idx_end_train = len(trainset)
            self.idx_start_train, self.idx_end_train = self.check_range(self.idx_start_train, self.idx_end_train)
            self.trainset = torch.utils.data.dataset.Subset(trainset, range(self.idx_start_train, self.idx_end_train))

            # validation data
            if self.idx_end_valid > len(valset): # need to add this. Bug?
                self.idx_end_valid = len(valset)
            self.idx_start_valid, self.idx_end_valid = self.check_range(self.idx_start_valid, self.idx_end_valid)
            self.valset = torch.utils.data.dataset.Subset(valset, range(self.idx_start_valid, self.idx_end_valid))

            # test data
            testset = torchvision_data(root=self.data_dir, train=False, download=True, transform=self.transform)
            if self.idx_end_test > len(testset):
                self.idx_end_test = len(testset)
            self.idx_start_test, self.idx_end_test = self.check_range(self.idx_start_test, self.idx_end_test)
            self.testset = torch.utils.data.dataset.Subset(testset, range(self.idx_start_test, self.idx_end_test))

    # return dataloader for distributed training before preparation
    def get_dataloader(self, batch_size_train, batch_size_test):
        train_dataloader = DataLoader(self.trainset, batch_size=batch_size_train)
        test_dataloader = DataLoader(self.testset, batch_size=batch_size_test)

        return train_dataloader, test_dataloader

    # return trainset, valset, testset in numpy array
    def get_data_numpy(self):
        dict_trainset = {'img':np.array([img.numpy()[0]*raw_amp_scale for img,_ in self.trainset]), 
                         'label':np.array([label for _,label in self.trainset])}
        dict_valset = {'img':np.array([img.numpy()[0]*raw_amp_scale for img,_ in self.valset]),
                       'label':np.array([label for _,label in self.valset])}
        dict_testset = {'img':np.array([img.numpy()[0]*raw_amp_scale for img,_ in self.testset]),
                        'label':np.array([label for _,label in self.testset])}

        return dict_trainset, dict_valset, dict_testset

    # convert torch tensor to numpy array
    def torch_tensor_to_numpy(self, img, label):
        img = img.numpy()*raw_amp_scale
        label = label.numpy()

        return {'img':img[:,0,:,:], 'label':label}
