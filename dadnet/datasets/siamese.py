class Siamese(Dataset):


    def __init__(self, *args):
    
       #init data here
    
    def __len__(self):
        return   #length of the data

    def __getitem__(self, idx):
        #get images and labels here 
        #returned images must be tensor
        #labels should be int 
        return img1, img2 , label1, label2