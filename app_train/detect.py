import numpy as np
import cv2
import torch
import glob as glob

from model import create_model

print("Torch version:", torch.__version__)

#MODEL_PATH = '../outputs/model100.pth'
#MODEL_PATH = 'train.pth'
MODEL_PATH =  'model_ctest1_craze3.pth'#'model_vim100_4e_4b.pth'
DIR_TEST = 'test_data'

detection_threshold = 0.7
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print("DV?",device)
device = torch.device('cpu')

# CLASSES = [
#     'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
# ]
# CLASSES = [
#     'background', 'CAT'
# ]

# CLASSES = [
#     'background', 'COOKING_PRODUCTS', 'DEODORANTS', 'DISHWASH', 'FABRIC_CONDITIONERS', 
#     'FABRICS_CLEANING', 'FACE', 'HAND_BODY_CARE', 'HOUSEHOLD_CLEANING','LEAF_INSTANT_TEA'
#     'MOUTHWASH_OTHER_ORAL', 'OUT_OF_HOME_ICE_CREAM', 'POSM_DISPLAY', 'SIP', 'SKIN_CLEANSING',
#     'STYLING','THI_DUA', 'TOOTHBRUSH', 'TOOTHPASTE', 'WASH_CARE', 'OTHER'
# ]

'''CLASSES = [
    'background', 'BanChaiRang_CLOSEUP', 'BanChaiRang_PS', 'Giattay_OMO', 'Giattay_SURF', 'ChamSocDaMat_DOVE', 'ChamSocDaMat_HAZELINE', 'ChamSocDaMat_PONDS', 'ChamSocDaMat_SIMPLE', 'ChamSocThanThe_DOVE', 'ChamSocThanThe_HAZELINE', 'ChamSocThanThe_LOVEBEAUTYANDPLANET','ChamSocThanThe_VASELINE','ChamSocToc_CLEAR','ChamSocToc_DOVE'
]'''

# CLASSES = [
#     'background','BanChaiRang_PS','Giattay_OMO','Giattay_SURF','ChamSocToc_CLEAR','ChamSocToc_DOVE','ChamSocToc_LIFEBUOY','ChamSocToc_TRESEMME','KemDanhRang_PS','Giattay_COMFORT','NuocRuaChen_SUNLIGHT','NuocXaVai_COMFORT','TayRuaBeMat_SUNLIGHT','TayRuaBeMat_VIM','TayRuaNhaVeSinh_VIM','VeSinhThanThe_LIFEBUOY','ChamSocToc_Xmen','ChamSocToc_Head&Shoulders','NuocXaVai_Downy','NuocSucMieng_LISTERINE'
# ]

# CLASSES = [
#     'background','COMFORT','OMO','SUNLIGHT', 'SURF', 'KDR_PS', 'NVS_VIM'
# ]

CLASSES = [
    'background','NXV_COMFORT','BG_OMO','NRC_SUNLIGHT','SURF','KDR_PS','NVS_VIM','DG_CLEAR','CST_DOVE','DG_SUNSILK','DG_LIFEBUOY','CST_TRESEME','NLS_SUNLIGHT','ST_LIFEBUOY','ST_HAZELINE','SRM_HALELINE','BCR_PS_LE','BCR_PS_KHOI','CST_XMEN','CST_HEADAS','CST_REJOICE','CST_PENTINE','SRM_BIORE','NSM_LISTERINE','KDR_CLOSEUP','BN_KNOR','DG_ROMANO','NKM_NIVEA','CSD_VASELINE','KDR_COLGATE','NKM_XMEN','KDR_SENSODYZE','CSD_PONDS','KDR_SENSITIVE','SRM_PONDS','CSD_NIVEA','KDR_DOREEN','XBC_LIFE_BOUY','OTHER_LIFE_BOUY','NLK_SUNLIGHT','NTT_JAVEL','NTL_247','NXV_Downy','BG_ARIEL','B_CHOCOBIE','ST_GERVENNE','DG_PATENE','ST_Lapetal','SUA_MILO','NU_TEA_PLUS','OISHI_SLIDE','BG_LIX','BG_ABA','DG_PALMOLIVE','DG_DUOCLIEU','NRC_SACH','NRC_LIX','NRC_SIEUSACH','ST_LUX','ST_OLAY','T_CARYN','CM_SOFFELL','ST_ISANA','ST_ENCHANTEUR','ST_E100','DG_OLIV','DG_HEADS&SHOUDERS','DG_REJOICE'
]


print(f'Labels Class Pred: {len(CLASSES)}')

model = create_model(num_classes=len(CLASSES)).to(device)
pth = torch.load(MODEL_PATH,map_location=device)
model.load_state_dict(pth)
model.eval()

print(f"Models Loaded {MODEL_PATH}!!!! @@@")

# test_images = glob.glob(f"{DIR_TEST}/*")
# print(f"Test instances: {len(test_images)}")

def DectImg2Brand(img_test: str):

    #img_test = f"{DIR_TEST}/df.jpg"
    
    brand_labels = []


    image_name = img_test.split('/')[-1].split('.')[0]
    image = cv2.imread(img_test)


 

    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(float)
    #image = np.array(image).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
        
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    if len(outputs[0]['boxes']) != 0:
        print('has box')
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        
        #print(boxes)
        print(scores)

        boxes = boxes[scores >= detection_threshold].astype(int)
        draw_boxes = boxes.copy()

        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        print(pred_classes)
        print(boxes)
        
        d = 0
        
        for j, box in enumerate(draw_boxes):
            d = d+1
            if scores[d-1] > detection_threshold:
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j], 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 0), 
                            2, lineType=cv2.LINE_AA)
                detect_label = pred_classes[j]
                
                print(f"detect_label: {detect_label}")
                
                if detect_label not in brand_labels:
                    brand_labels.append(detect_label)
                print(f'has {pred_classes[j]} {d}' )
                print('----')
                print(scores[d-1])
                print('-------')
        
        
        cv2.imwrite(f"test_predictions/{image_name}.jpg", orig_image,)
    else:
        print("nox box")
    
    print("detect_brand_labels___",brand_labels)
    return brand_labels
    
DectImg2Brand("test_img/test_r8.jpg")