
folder = "data_proc/d_35"
#folder = "data_proc/test2"

train_data_dir = f"{folder}/images"
train_coco = f"{folder}/annotations/instances_default.json"
#train_coco = f"{folder}/annotations/d_35_clean_2cat.json"


train_batch_size = 1


train_shuffle_dl = True
num_workers_dl = 4


num_classes = 65 # BlueForce
num_epochs = 10

lr = 0.0005
momentum = 0.9
weight_decay = 0.0005
