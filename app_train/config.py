
folder = "data_proc/nutri"
#folder = "data_proc/test2"

#train_data_dir = f"{folder}/images_mini"
#train_coco = f"{folder}/annotations/bf_45cat.json"
#train_coco = f"{folder}/annotations/d_35_clean_2cat.json"
#train_coco = f"{folder}/annotations/d_ken_clean4.json"

train_data_dir = f"{folder}/images"
train_coco = f"{folder}/annotations/anno_train.json"

train_batch_size = 1


train_shuffle_dl = True
num_workers_dl = 10


#num_classes = 68 umer
#num_classes =  46 # BlueForce #106
#num_classes = 39 # Ken 39 60
num_classes = 96 # Nutri
num_epochs = 10

lr = 0.0005
momentum = 0.9
weight_decay = 0.0005


# KEN NOTE
# 46 labels - 2048 images - 32878 anno


# NUTRI NOTE
# 95 +1 labels - 2347 images - 98184