import numpy as np
import cv2
import torch
import glob as glob
import uuid
from urllib.request import urlopen
from model import create_model
from contextlib import asynccontextmanager
from fastapi import FastAPI


#model_PATH =  'model/model_cat_brand_v1.pth' # UMER
#model_PATH =  'model/model_bf_45cat_12.pth' # BLUEFORCE
#model_PATH =  'model/model_ken_134_4.pth' # KEN
model_PATH =  'model/model_nutri2.pth'
#model_PATH =  'model/model_vnmk_mini60.pth'
#model_PATH = 'model/model_ken_mini.pth'

DIR_TEST = 'test_data'

detection_threshold = 0.7
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print("DV?",device)
device = torch.device('cpu')



# CLASSES = ['0_bg' ,'TIGER_CRYTAL_CHAI','STRONGBOW_LON','BIA_VIET_LON','KEN_LON','BIVINA_LON','TIGER_LON','STRONGBOW_CHAI','TIGER_SOJU_LON','BIVINA_EXP_LON','TIGER_CRYTAL_LON','LARUE_LON','Tiger_Platinum_Lon','LARUE_SPECIAL_LON','KEN_0_LON','KEN_SIL_COOL_LON','TIGER_LOC','STRONGBOW_LOC','KEN_SIL_LON','SAIGON_LAGER_THUNG','Budweiser_Thung','KEN_CHAI','BIA_VIET_CHAI','TIGER_CRYTAL_THUNG','Hanoi_Thung','HUDA_THUNG','TIGER_COOL_LON','Huda_Lon','SAIGON_LAGER_LON','ken_0_loc','333_lon','KEN_THUNG','Edelweiss_Lon','SAPPORO_LON','TIGER_THUNG','Larue_Smooth_Lon','LARUE_CHAI','KEN_SIL_CHAI','TIGER SOJU_LOC','EDELWEISS_LOC','LARUE_THUNG','TIGER_CHAI','HA_NOI_PREMIUM_THUNG','TIGER_PLATINUM_LOC','Bivina_Export_Chai','HEINEKEN_LOC','TIGER_CRYSTAL_LOC','HEINEKEN SILVER_LOC','333_THUNG','SAIGON_KET','Ruby_thung','SAPPORO_PREMIUM_THUNG','TIGER_SOJU_THUNG','Bia Viet_Ket','TIGER_CRYTAL_KET','Ken_0_Thung','Saigon_export_premium_thung','TIGER_CRYTAL_COOL_LON','KEN_SIL_THUNG','Sai Gon Special_Thung','Ha_Noi_Lon','1664_BLANC_LON','Carlsberg_Lon','LARUE_SPECIAL_THUNG','Tiger_Soju_Chai','KEN_SIL_COOL_THUNG','Sai Gon Special_ket','Tiger Platinum_Thung','Huda_Ket','BIVINA_EXP_THUNG','budweiser_lon','LARUE_KET','BIA_VIET_THUNG','BIVINA_THUNG','Saigon Chill_Thung','TRUC_BACH_LON','LACVIET_THUNG','QUI_NHON_KET','SAIGON_CHILL_LON','TUBORG_LON','SAIGON_SPECIAL_LOC','Tiger_Crystal_Cool_Thung','HALIDA_THUNG','Tiger_Cool_Thung','CARLSBERG_THUNG','Truc Bach_Thung','HANOI_PREMIUM_LON','Tiger_Ket','Ken_Sil_Ket','EDELWEISS_CHAI','Larue Smooth_Thung','STRONGBOW_THUNG','SAIGON_SPECIAL_LON','1664_BLANC_THUNG','Heineken_Ket','333_LOC','DAIVIET_LON','TUBORG_THUNG','BUDWEISER_LOC','Corona_Extra_Chai','Somersby_Lon','SAIGON CHILL_LOC','SAPPORO_LOC','HOT_BIAS_THUNG','Tiger_Soju_Ket','SAIGON_CHILL_KET','SAIGON LAGER_LOC','TUBORG_ICE_THUNG','saigon_export_thung','G8_PREMIUM_THUNG','DUNG QUAT_KET','1664_BLANC_LOC','LEFFE_CHAI','BECKS_LON','HALIDA_LON','BECKS_THUNG','BIA_VIET_LOC','HEINEKEN_COOLPACK_LON','Sai Gon Special_Chai','HOEGAARDEN_LON','BECKS_ICE_LOC','SAIGON_GOLD_LON','TUBORG_KET','BECKS_ICE_THUNG','HANOI_LOC','Sapphire_Thung','saigon_export_premium_lon','huda_ice_twist_thung','TUBORG_ICE_LON','HOEGAARDEN_THUNG','LARUE_LOC','TUBORG_LOC','1664 BLANC_CHAI','SAOVANG_THUNG']


# CLASSES = ['0_bg' ,'KEN_LON','TIGER_CRYTAL_CHAI','STRONGBOW_LON','BIA_VIET_LON','BIVINA_LON','TIGER_LON','STRONGBOW_CHAI','TIGER_SOJU_LON','KEN_0_LON','BIVINA_EXP_LON','TIGER_CRYTAL_LON','LARUE_LON','Tiger_Platinum_Lon','LARUE_SPECIAL_LON','KEN_SIL_COOL_LON','TIGER_LOC','STRONGBOW_LOC','KEN_SIL_LON','SAIGON_LAGER_THUNG','Budweiser_Thung','KEN_CHAI','BIA_VIET_CHAI','TIGER_CRYTAL_THUNG','Hanoi_Thung','HUDA_THUNG','TIGER_COOL_LON','Huda_Lon','SAIGON_LAGER_LON','ken_0_loc','333_lon','KEN_THUNG','Edelweiss_Lon','SAPPORO_LON','TIGER_THUNG','Larue_Smooth_Lon','LARUE_CHAI','KEN_SIL_CHAI','TIGER SOJU_LOC','EDELWEISS_LOC','LARUE_THUNG','TIGER_CHAI','HA_NOI_PREMIUM_THUNG','TIGER_PLATINUM_LOC','Bivina_Export_Chai','HEINEKEN_LOC','TIGER_CRYSTAL_LOC','HEINEKEN SILVER_LOC','333_THUNG','SAIGON_KET','Ruby_thung','SAPPORO_PREMIUM_THUNG','TIGER_SOJU_THUNG','Bia Viet_Ket','TIGER_CRYTAL_KET','Ken_0_Thung','Saigon_export_premium_thung','TIGER_CRYTAL_COOL_LON','KEN_SIL_THUNG','Sai Gon Special_Thung','Ha_Noi_Lon','1664_BLANC_LON','Carlsberg_Lon','LARUE_SPECIAL_THUNG','Tiger_Soju_Chai','KEN_SIL_COOL_THUNG','Sai Gon Special_ket','Tiger Platinum_Thung','Huda_Ket','BIVINA_EXP_THUNG','budweiser_lon','LARUE_KET','BIA_VIET_THUNG','BIVINA_THUNG','Saigon Chill_Thung','TRUC_BACH_LON','LACVIET_THUNG','QUI_NHON_KET','SAIGON_CHILL_LON','TUBORG_LON','SAIGON_SPECIAL_LOC','Tiger_Crystal_Cool_Thung','HALIDA_THUNG','Tiger_Cool_Thung','CARLSBERG_THUNG','Truc Bach_Thung','HANOI_PREMIUM_LON','Tiger_Ket','Ken_Sil_Ket','EDELWEISS_CHAI','Larue Smooth_Thung','STRONGBOW_THUNG','SAIGON_SPECIAL_LON','1664_BLANC_THUNG','Heineken_Ket','333_LOC','DAIVIET_LON','TUBORG_THUNG','BUDWEISER_LOC','Corona_Extra_Chai','Somersby_Lon','SAIGON CHILL_LOC','SAPPORO_LOC','HOT_BIAS_THUNG','Tiger_Soju_Ket','SAIGON_CHILL_KET','SAIGON LAGER_LOC','TUBORG_ICE_THUNG','saigon_export_thung','G8_PREMIUM_THUNG','DUNG QUAT_KET','1664_BLANC_LOC','LEFFE_CHAI','BECKS_LON','HALIDA_LON','BECKS_THUNG','BIA_VIET_LOC','HEINEKEN_COOLPACK_LON','Sai Gon Special_Chai','HOEGAARDEN_LON','BECKS_ICE_LOC','SAIGON_GOLD_LON','TUBORG_KET','BECKS_ICE_THUNG','HANOI_LOC','Sapphire_Thung','saigon_export_premium_lon','huda_ice_twist_thung','TUBORG_ICE_LON','HOEGAARDEN_THUNG','LARUE_LOC','TUBORG_LOC','1664 BLANC_CHAI','SAOVANG_THUNG']

#CLASSES = ['0_bg' ,'KEN_LON','TIGER_CRYTAL_CHAI','STRONGBOW_LON','BIA_VIET_LON','BIVINA_LON','TIGER_LON','STRONGBOW_CHAI','TIGER_SOJU_LON','KEN_0_LON','BIVINA_EXP_LON','TIGER_CRYTAL_LON','LARUE_LON','Tiger_Platinum_Lon','LARUE_SPECIAL_LON','KEN_SIL_COOL_LON','TIGER_LOC','STRONGBOW_LOC','KEN_SIL_LON','SAIGON_LAGER_THUNG','Budweiser_Thung','KEN_CHAI','BIA_VIET_CHAI','TIGER_CRYTAL_THUNG','Hanoi_Thung','HUDA_THUNG','TIGER_COOL_LON','Huda_Lon','SAIGON_LAGER_LON','ken_0_loc','333_lon','KEN_THUNG','Edelweiss_Lon','SAPPORO_LON','TIGER_THUNG','Larue_Smooth_Lon','LARUE_CHAI','KEN_SIL_CHAI','TIGER SOJU_LOC','EDELWEISS_LOC','LARUE_THUNG','TIGER_CHAI','HA_NOI_PREMIUM_THUNG','TIGER_PLATINUM_LOC','Bivina_Export_Chai','HEINEKEN_LOC','TIGER_CRYSTAL_LOC']

CLASSES = ['0_bg','3_TIGER','2_KEN_LON','36_TIGER_TXT','5_BIA_VIET','4_STRONGBOW','15_BIA_HUDA','33_BIA_SAIGON_TXT','6_LARUE','10_BIA_333','9_BIA_SAIGON','8_BIVINA','37_HEINEKEN_TXT','11_SAPPORO','14_CARLSBERG','35_TUBORG_TXT','13_BIA_HANOI','7_EDELWEISS','24_BIA_TUBORG','1_BIA_KEN','38_SAPPORO_TXT','34_BIA_VIET_TXT','44_BUDWEISER_TXT','42_BIA_HANOI_TXT','12_RUBY','23_BUDWEISER','19_BIA_TRUCBACH','39_LARUE_TXT','16_1664_BLANC_LON','40_BLANC_TXT','42_BECK_TXT','26_BIA_BECK','27_BIA_BECK_ICE','43_BECK_ICE_TXT','25_CORONA','18_BIA_HOI_HANOI','17_1664_BLANC_CHAI','28_SAPPHIRE','41_SAPPHIRE_TXT']

#CLASSES = ['0_bg','3_VNM_100','4_VNM_Proby','8_VNM_Ongtho','9_VNM_NgoiSao','C1_TH_MILK','13_VNM_Susu','10_VNM_GrowPlus','12_VNM_Alpha','22_VNM_Colos','7_VNM_ADM','16_VNM_SuperNut','1_VNM_Suachua','11_VNM_GreenFarm','21_VNM_YokoGold','2_VNM_Opti_Gold','15_VNM_HERO','17_VNM_Sure','26_VNM_Ridielac','C2_Dalat_Milk','C7_Milo','6_VNM_SuaDinhDuong','C5_Yakult','C16_Nutri_NUVI','20_VNM_Flex','C10_PhomaiCBC','C15_DutchLady','C4_Betagen','VNM_TAILOC','18_VNM_Organic','C11_VFRESH','Nutifood_GrowPlus','C3_Yomost','KUN','5_VNM_Phomai','Nutifood_NutiMilk','C12_Meadow','19_VNM_SuperSoy','23_VNM_YoMilk','Nestle_NAN','C9_Monte','Abbott_Glucena','KENKO_HARU','C8_Hoff','VNM_SDNTuoi','TH_TRUE_YOGURT','C6_Promess','Nutifood_nuti','PEDIA_KENJI','VNM_HAPPYSTAR','C13_LONGTHANH_Milk','Ovaltine','24_VNM_Mama','C14_HIPP','MeiJi','VNM_OPTI','27_VNM_STAR','Nuvi_TXT','Nutricare_Metacare','25_VNM_GROW']

# CLASSES = ['0_bg','3_VNM_100','10_VNM_GrowPlus','C1_TH_MILK','Nestle_NAN','12_VNM_Alpha','4_VNM_Proby','22_VNM_Colos','21_VNM_YokoGold','26_VNM_Ridielac','2_VNM_Opti_Gold','13_VNM_Susu','16_VNM_SuperNut','8_VNM_Ongtho','9_VNM_NgoiSao','C2_Dalat_Milk','17_VNM_Sure','11_VNM_GreenFarm','ABT_GROWABT','7_VNM_ADM','15_VNM_HERO','C5_Yakult','C7_Milo','1_VNM_Suachua','6_VNM_SuaDinhDuong','Friso','ABT_SIMILAC','C4_Betagen','C15_DutchLady','ABT_ENFA','C16_Nutri_NUVI','C10_PhomaiCBC','Nutifood_GrowPlus','ABT_NUTIFOOD_GROWPLUS','C3_Yomost','KUN','ABT_C_VNM_OPTIMUM','ABT_C_MEJI','20_VNM_Flex','VNM_TAILOC','C8_Hoff','ABT_C_VNM_ALPHA','ABT_PEDIASURE','C9_Monte','Ensure','Anlene','18_VNM_Organic','23_VNM_YoMilk','C14_HIPP','5_VNM_Phomai','C11_VFRESH','VNM_SUACHUA_NEW','ABT_C_COLOS_BABY','C13_LONGTHANH_Milk','Similac','Abbott_Grow','Abbott_Glucena','ABT_C_HIKID','C12_Meadow','ABT_SIMILAC_HOP','TH_TRUE_YOGURT','Nutifood_NutiMilk','19_VNM_SuperSoy','ABT_PEDIASURE_HOP','MeiJi','Nutifood_nuti','ABT_C_BACKMORE','Nutricare_Metacare','ABT_C_APTAMILK','PEDIA_KENJI','C6_Promess','ABT_ENSURE','KENKO_HARU','VNM_SDNTuoi','Cerelac','VNM_PROBY_NEW','24_VNM_Mama','VNM_HAPPYSTAR','Ovaltine','Nuvi_TXT','VNM_SUACHUA_NHADAM_NEW','ABT_C_MORINAGA','Nutifood_Famna','VNM_OPTI','27_VNM_STAR','ABT_GLUCENA','ABT_APTAKID','ABT_C_HUMANA','Anmum','ABT_C_ENFA_HOP','25_VNM_GROW','VNM_SUACHUA_DAU_NEW','VNM_SUACHUA_TRAICAY_NEW','ABT_C_VITADIARY_OGGI']


# CLASSES = ['0_bg','3_VNM_100','Nestle_NAN','10_VNM_GrowPlus','ABT_C_VNM_OPTIMUM','C1_TH_MILK','ABT_GROWABT','ABT_C_VNM_ALPHA','21_VNM_YokoGold','Friso','ABT_ENFA','ABT_SIMILAC','22_VNM_Colos','4_VNM_Proby','26_VNM_Ridielac','ABT_NUTIFOOD_GROWPLUS','13_VNM_Susu','12_VNM_Alpha','16_VNM_SuperNut','8_VNM_Ongtho','9_VNM_NgoiSao','17_VNM_Sure','C2_Dalat_Milk','2_VNM_Opti_Gold','ABT_C_MEJI','11_VNM_GreenFarm','C7_Milo','ABT_ENSURE','7_VNM_ADM','15_VNM_HERO','ABT_PEDIASURE','C5_Yakult','1_VNM_Suachua','6_VNM_SuaDinhDuong','C15_DutchLady','C4_Betagen','ABT_PEDIASURE_HOP','C16_Nutri_NUVI','ABT_C_APTAMILK','ABT_C_BACKMORE','ABT_SIMILAC_HOP','ABT_C_COLOS_BABY','C10_PhomaiCBC','C14_HIPP','ABT_C_HIKID','C3_Yomost','Anlene','KUN','20_VNM_Flex','C12_Meadow','ABT_C_MORINAGA','Nutifood_GrowPlus','VNM_TAILOC','C8_Hoff','ABT_GLUCENA','C9_Monte','C13_LONGTHANH_Milk','18_VNM_Organic','23_VNM_YoMilk','Nutricare_Metacare','5_VNM_Phomai','C11_VFRESH','VNM_SUACHUA_NEW','Nutifood_Famna','ABT_C_VNM_CANXI','Nutifood_NutiMilk','TH_TRUE_YOGURT','19_VNM_SuperSoy','ABT_C_HUMANA','Nutifood_nuti','ABT_APTAKID','MeiJi','Cerelac','C6_Promess','PEDIA_KENJI','VNM_PROBY_NEW','Abbott_Glucena','KENKO_HARU','Similac','VNM_SDNTuoi','Nuvi_TXT','24_VNM_Mama','Abbott_Grow','VNM_HAPPYSTAR','Ovaltine','ABT_C_ENFA_HOP','VNM_SUACHUA_NHADAM_NEW','Anmum','Ensure','27_VNM_STAR','ABT_C_NEWSTLE_BOOSTOPTIMUM','VNM_OPTI','25_VNM_GROW','ABT_C_VITADIARY_OGGI','VNM_SUACHUA_DAU_NEW','VNM_SUACHUA_TRAICAY_NEW']

CLASSES = ['0_bg','3_VNM_100','Nestle_NAN','10_VNM_GrowPlus','ABT_C_VNM_OPTIMUM','C1_TH_MILK','ABT_GROWABT','ABT_C_VNM_ALPHA','21_VNM_YokoGold','Friso','ABT_ENFA','ABT_SIMILAC','22_VNM_Colos','4_VNM_Proby','26_VNM_Ridielac','ABT_NUTIFOOD_GROWPLUS','13_VNM_Susu','12_VNM_Alpha','16_VNM_SuperNut','8_VNM_Ongtho','9_VNM_NgoiSao','17_VNM_Sure','C2_Dalat_Milk','2_VNM_Opti_Gold','ABT_C_MEJI','11_VNM_GreenFarm','C7_Milo','ABT_ENSURE','7_VNM_ADM','15_VNM_HERO','ABT_PEDIASURE','C5_Yakult','1_VNM_Suachua','6_VNM_SuaDinhDuong','C15_DutchLady','C4_Betagen','ABT_PEDIASURE_HOP','C16_Nutri_NUVI','ABT_C_APTAMILK','ABT_C_BACKMORE','ABT_SIMILAC_HOP','ABT_C_COLOS_BABY','C10_PhomaiCBC','C14_HIPP','ABT_C_HIKID','C3_Yomost','Anlene','KUN','20_VNM_Flex','C12_Meadow','ABT_C_MORINAGA','Nutifood_GrowPlus','VNM_TAILOC','C8_Hoff','ABT_GLUCENA','C9_Monte','C13_LONGTHANH_Milk','23_VNM_YoMilk','18_VNM_Organic','Nutricare_Metacare','VNM_SUACHUA_NEW','5_VNM_Phomai','C11_VFRESH','Nutifood_Famna','ABT_C_VNM_CANXI','Nutifood_NutiMilk','TH_TRUE_YOGURT','19_VNM_SuperSoy','ABT_C_HUMANA','Nutifood_nuti','ABT_APTAKID','MeiJi','VNM_PROBY_NEW','Cerelac','C6_Promess','PEDIA_KENJI','Abbott_Glucena','KENKO_HARU','Similac','VNM_SDNTuoi','Nuvi_TXT','24_VNM_Mama','Abbott_Grow','VNM_HAPPYSTAR','Ovaltine','VNM_SUACHUA_NHADAM_NEW','ABT_C_ENFA_HOP','Anmum','Ensure','27_VNM_STAR','ABT_C_NEWSTLE_BOOSTOPTIMUM','VNM_OPTI','25_VNM_GROW','ABT_C_VITADIARY_OGGI','VNM_SUACHUA_DAU_NEW','VNM_SUACHUA_TRAICAY_NEW']


# chay 2 tuan detect 8/8 - 22/8
# 


print(f'Labels Class Pred: {len(CLASSES)}')


dectectAPI = None

model_ai = {}
# test_images = glob.glob(f"{DIR_TEST}/*")
# print(f"Test instances: {len(test_images)}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    global dectectAPI
    # Load the ML model
    print("Torch version:", torch.__version__)
    model = create_model(num_classes=len(CLASSES)).to(device)
    pth = torch.load(model_PATH,map_location=device)
    model.load_state_dict(pth)
    model.eval()
    model_ai["DectectAPI"] = DectectAPI(model=model)
    print(f"models Loaded {model_PATH}!!!! @@@")
    yield
    # Clean up the ML self.models and release the resources
    model_ai.clear()


class DetectObject:
    brands = []
    img_out = ""
    photo_id = 0
    photo_date = 0
    photo_link = ""
    img_url = ""
    shop_code = ""

class DectectAPI:
    def __init__(self,model) -> None:
        self.model = model
    def DectImgUrl2Brand(self,img_url: str, img_name: str):
            #img_test = f"{DIR_TEST}/df.jpg"
        
        detect_object = DetectObject()
        brand_labels = []
        

        try:
            image_url = img_url
            resp = urlopen(image_url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR) # The image object

            # Optional: For testing & viewing the image
            #cv2.imshow('image',image)
            
            #key = cv2.waitKey(0)
            #print("done! get img_url")
        except Exception as ex:
            print('ex: ', ex)

        image_name = img_name
        
        #image = cv2.imread(img_test)


    

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
            outputs = self.model(image)
            
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:
            #print('has box')
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            
            #print(boxes)
            #print(scores)

            boxes = boxes[scores >= detection_threshold].astype(int)
            draw_boxes = boxes.copy()

            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            #print(pred_classes)
            #print(boxes)
            
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
                    
                    #print(f"detect_label: {detect_label}")
                    
                    if detect_label not in brand_labels:
                        brand_labels.append(detect_label)
                    #print(f'has {pred_classes[j]} {d}' )
                    #print('----')
                    #print(scores[d-1])
                    #print('-------')
            
            img_out = f"{uuid.uuid4().hex}_{image_name}"
            detect_object.img_out = img_out
            #cv2.imwrite(f"test_predictions/{img_out}", orig_image,)
            detect_object.img_out = "ok"
        else:
            print("nox box")
        
        detect_object.brands = brand_labels

        detect_object.img_url = img_url

        
        print(detect_object)
        return detect_object
        

    def DectImg2Brand(self,img_test: str):

        #img_test = f"{DIR_TEST}/df.jpg"
        
        detect_object = DetectObject()
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
            outputs = self.model(image)
            
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
                    
                    #print(f"detect_label: {detect_label}")
                    
                    if detect_label not in brand_labels:
                        brand_labels.append(detect_label)
                    #print(f'has {pred_classes[j]} {d}' )
                    #print('----')
                    #print(scores[d-1])
                    #print('-------')
            
            img_out = f"{image_name}_{uuid.uuid4().hex}.jpg"
            detect_object.img_out = img_out
            #cv2.imwrite(f"test_predictions/{img_out}", orig_image,)
        else:
            print("nox box")
        
        detect_object.brands = brand_labels
        
        print("detect_brand_labels___",detect_object)
        return detect_object

    def DectBrand(self,img_test: str):

        #img_test = f"{DIR_TEST}/df.jpg"
        
        detect_object = DetectObject()
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
            outputs = self.model(image)
            
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:

            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            


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
                    

                    
                    if detect_label not in brand_labels:
                        brand_labels.append(detect_label)

            
            #img_out = f"{image_name}_{uuid.uuid4().hex}.jpg"
            #detect_object.img_out = img_out
            #cv2.imwrite(f"test_predictions/{img_out}", orig_image,)
        else:
            print("nox box")
        
        detect_object.brands = brand_labels
        
        #print("detect_brand_labels___",detect_object)
        return detect_object
        
    #DectImg2Brand("test_img/test_r8.jpg")

    #DectImgUrl2Brand("https://webhub.acacy.com.vn/storage/images/view?guid=241E578DF36B96C4ACFE48E6DA0133E5",'test')