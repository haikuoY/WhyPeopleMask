import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from tqdm import tqdm
import json
import random

# class_list = ['lottery', 'weibo_page', 'mouse', 'photo', 'low_body_part', 'file', 'pilliow', 'passerby', 'pill', 'swimming_ring', 'ID_card', 'slipper', 'bike', 'musical_instruments', 'female', 'plate', 'blackboard', 'sign', 'ID', 'face', 'hand', 'bowl', 'toy', 'Trump', 'literal_part', 'flower', 'glass', 'light', 'bed', 'marriage_card', 'stool', 'screen', 'hot_pot', 'table', 'prizewinner', 'bottle', 'baby', 'photo_part', 'mask', 'children', 'Express_label', 'visa', 'chair', 'bill', 'car', 'male', 'drive_license', 'sculpture', 'laptop', 'bassinet', 'cat', 'drink', 'video_part', 'address', 'road_sign', 'suitcase', 'watch', 'eye', 'wedding_dress', 'repro_part', 'person', 'towel', 'mouth', 'marriage_license', 'dog', 'bag', 'mobile', 'window']

class_list = ['airport', 'beach', 'Express_label', 'ID', 'ID_card', 'TV', 'address', 'age', 'baby', 'bag', 'balance', 'bank_card', 'bar_code', 'bassinet', 'battledore', 'bed', 'bedroom', 'belly_button', 'bike', 'birth', 'blackboard', 'body', 'book', 'bottle', 'bowl', 'business_card', 'cake', 'calendar', 'car', 'card_number', 'cat', 'chair', 'chat_shot', 'children', 'classroom', 'club_card', 'computer', 'correspondent_part', 'cup', 'date', 'dog', 'drink', 'drink_label', 'drive_license', 'elevator', 'em_card', 'exam_card', 'expenditure', 'face', 'female', 'file', 'flower', 'food', 'game_shot', 'glass', 'glasses', 'grade', 'gym', 'hand', 'hat', 'head_portrait', 'indoor', 'in_car', 'keyboard', 'kitchen', 'lamp', 'laptop', 'library', 'literal_part', 'living_room', 'logo', 'lottery', 'male', 'mall', 'marriage_license', 'mask', 'medical_record', 'medical_report', 'microphone', 'mobile', 'motorcycle', 'mouse', 'musical_instruments', 'name', 'nation', 'nickname', 'office', 'onlinemeeting', 'outside', 'park', 'party', 'passageway', 'passerby', 'passport', 'passport_number', 'pen', 'person', 'phone_number', 'photo', 'photo_part', 'pill', 'pillow', 'plane', 'plate', 'plate_number', 'playground', 'pot', 'qrcode', 'repro_part', 'restaurant', 'restroom', 'road_sign', 'rostrum', 'school_report', 'screen', 'screen_shot', 'sculpture', 'sex', 'shop', 'shower_room', 'sign', 'station', 'station_info', 'stool', 'street', 'student_number', 'subway', 'suitcase', 'sunglasses', 'swimming_pool', 'swimming_ring', 'table', 'table_top', 'ticket', 'toiletries', 'towel', 'toy', 'train', 'train_number', 'video_call', 'video_part', 'watch', 'wedding_dress', 'window', 'work_card', 'belly', 'balloon', 'amusement_park', 'campus', 'dormitory']

Scene_list = ['airport', 'amusement_park', 'beach', 'bedroom', 'campus', 'classroom', 'dormitory', 'gym', 'library', 'living_room', 'indoor', 'in_car', 'kitchen', 'mall', 'onlinemeeting', 'station', 'office', 'outside', 'park', 'party', 'passageway', 'playground', 'restaurant', 'restroom', 'rostrum', 'screen_shot', 'shop', 'shower_room', 'station', 'street', 'swimming_pool', 'subway', 'table_top']

text_list = ['ID', 'address', 'age', 'balance', 'birth', 'date', 'expenditure', 'grade', 'literal_part', 'medical_record', 'name', 'nation', 'nickname', 'passport_number', 'person_info', 'plate_number', 'phone_number', 'plate_number', 'sex', 'station_info', 'student_number', 'train_number', 'video_call', ]

object_list = ['Express_label', 'ID_card', 'TV', 'baby', 'bag', 'bar_code', 'bassinet', 'battledore', 'bed', 'belly_button', 'bike', 'blackboard', 'body', 'book', 'bottle', 'bowl', 'business_card', 'cake', 'calendar', 'car', 'cat', 'chair', 'chat_shot', 'children', 'club_card', 'computer', 'correspondent_part', 'cup', 'dog', 'drink', 'drink_label', 'drive_license', 'elevator', 'em_card', 'exam_card', 'face', 'female', 'file', 'flower', 'food', 'game_shot', 'glass', 'hand', 'hat', 'head_portriat', 'keyboard', 'lamp', 'laptop', 'logo', 'lottery', 'male', 'marriage_license', 'mask', 'medical_report', 'microphone', 'mobile', 'motorcycle', 'mouse', 'musical_instruments', 'passerby', 'passport', 'pen', 'person', 'photo', 'photo_part', 'pill', 'pillow', 'plane', 'plate', 'pot', 'qrcode', 'repro_part', 'road_sign', 'school_report', 'screen', 'screen_shot', 'sculpture', 'sign', 'stool', 'suitcase', 'sunglass', 'swimming_ring', 'table', 'ticket', 'toiletries', 'towel', 'toy', 'train', 'watch', 'wedding_dress', 'window', 'work_card', 'belly', 'balloon']
class Graph():
    def __init__(self,adj_matrix, node, label, node_name, file_name, primary_key):
        self.adj_matrix, self.node, self.label, self.node_name = adj_matrix, node, label, node_name
        self.file_name = file_name
        self.primary_key = primary_key
    def printinfo(self):
        print("adj_matrix\n", self.adj_matrix) 
        # print("node", self.node)
        print("label", self.label)
        print("node_name\n", self.node_name)

# class GraphList():
#     def __init__(self, adj_matrix_list, node_list, label_list):
#         self.adj_matrix_list, self.node_list, self.label_list = adj_matrix_list, node_list, label_list
        
def load_class_dict(class_list):
    class_dict = {}
    l = len(class_list)
    for i in tqdm(range(0,l)):
        # onehot_label = np.zeros(shape=(l,))
        onehot_label = [0 for n in range(0,l)]
        onehot_label[i] = 1
        class_dict[class_list[i]] = onehot_label
    return class_dict

def generate_onehot_class(path): #返回一个字典
    xml_list = []
    # lable_list = []
    # xml_file_list = os.listdir(path)
    # xml_file_list.sort()
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # label = []
        for member in root.findall('object'):
            if member[0].text.endswith('_p'):
                name = member[0].text[:-2]
                # label.append(1)
            else:
                name = member[0].text
                # label.append(0)
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     name,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    #     lable_list.append(label)
    # label_np = np.array(lable_list)
    # np.save('dataset_classified\\test3\\test_label', label_np) 
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    class_list = xml_df["class"].to_list()
    class_list = list(set(class_list))
    class_list.sort()
    print(class_list)
    class_dict = {}
    l = len(class_list)
    for i in range(0,l):
        # onehot_label = np.zeros(shape=(l,))
        onehot_label = [0 for n in range(0,l)]
        onehot_label[i] = 1
        class_dict[class_list[i]] = onehot_label
    return class_list, class_dict

def bounding_box_overlap(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2):
    if(xmin1 >= xmax2) or (xmin2 >= xmax1) or (ymin1 >= ymax2) or (ymin2 >= ymax1):
        return 0
    area1 = (xmax1-xmin1)*(ymax1-ymin1)
    area2 = (xmax2-xmin2)*(ymax2-ymin2)
    inter_col = min(ymax2, ymax1) - max(ymin1, ymin2)
    inter_row = min(xmax2, xmax1) - max(xmin1, xmin2)
    inter_area = inter_row * inter_col
    # if area1 == inter_area:
    #     return inter_area/area1
    # else:
    #     return 0
    # print(area1,area2,inter_area)
    # return inter_area / (area1 + area2 - inter_area)
    return inter_area / area1

def xml_to_graph(path,class_dict,count): # 计算overlap
    # adj_matrix_list = []
    # node_list = []
    # lable_list = []
    Graph_list = []
    # count = 0
    a = 0
    b = 0
    file_list = glob.glob(path + '/*.xml')
    file_list.sort()
    privacy_scene = []
    for xml_file in file_list:
        # print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        node_vector = []
        # center_vector = []
        label_vector = []
        name_list = []
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)
        for member in root.findall('object'):
            a += 1
            if member[0].text == 'lottery':
                member[0].text = 'lottery_p'
            if member[0].text.endswith('_p'):
                name = member[0].text[:-2]
                b += 1
                if name in Scene_list:
                    privacy_scene.append(name)
                # label_vector.append([0,1])
                # x1=random.uniform(0,1)
                # if(x1<=0.9):
                #     label_vector.append(1)
                # else:
                #     label_vector.append(0)
                label_vector.append(1)
            else:
                name = member[0].text
                # label_vector.append([1,0])
                # x1=random.uniform(0,1)
                # if(x1<=0.9):
                #     label_vector.append(0)
                # else:
                #     label_vector.append(1)
                label_vector.append(0)
            name_list.append(name)
            if name == 'price':
                name = 'balance'
            if name == 'shoes':
                name = 'toiletries'
            if name == 'email':
                name = 'phone_number'
            if name == 'cap':
                name = 'hat'
            
            class_embedding = class_dict[name]
            
            x1 = int(member[4][0].text)/width
            y1 = int(member[4][1].text)/height
            x2 = int(member[4][2].text)/width
            y2 = int(member[4][3].text)/height
            center = [x1, y1, x2, y2, (x1+x2)/2, (y1+y2)/2]
            # center_vector.append(center)
            area = [(x2-x1)*(y2-y1)]
            ratio = [(x2-x1)*width/((x2-x1)*width+(y2-y1)*height)]
            node = class_embedding + center + ratio + area
            node_vector.append(node)
        adj_matrix = np.zeros((len(node_vector), len(node_vector)))
        i = 0
        # print(name_list,adj_matrix.shape)
        for mem1 in root.findall('object'):
            xmin1 = int(mem1[4][0].text)
            ymin1 = int(mem1[4][1].text)
            xmax1 = int(mem1[4][2].text)
            ymax1 = int(mem1[4][3].text)
            j = 0
            for mem2 in root.findall('object'):
                xmin2 = int(mem2[4][0].text)
                ymin2 = int(mem2[4][1].text)
                xmax2 = int(mem2[4][2].text)
                ymax2 = int(mem2[4][3].text)
                dis = bounding_box_overlap(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2)
                adj_matrix[i][j] = dis
                j += 1
            i += 1
        label_np = np.array(label_vector)
        node_np = np.array(node_vector)
        G = Graph(adj_matrix, node_np, label_np, name_list, xml_file, primary_key= count)

        Graph_list.append(G)
        count += 1
        
    # print(a,b)
    print(list(set(privacy_scene)))
    return Graph_list,count

def json_to_graph(folder_path, class_dict, count):
        # adj_matrix_list = []
    # node_list = []
    # lable_list = []
    Graph_list = []
    # count = 0
    file_list = os.listdir(folder_path)
    # print(file_list)
    for file_name in file_list:
        # print(file_name)
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)
        f = open(file_path,'r')
        d = json.load(f)
        node_vector = []
        # center_vector = []
        label_vector = []
        name_list = []
        # print(file_name)
        width = d['asset']['size']['width']
        height = d['asset']['size']['height']        
        for member in d['regions']:
            if member['tags'][0].endswith('_p'):
                name = member['tags'][0][:-2]
                # label_vector.append([0,1])
                label_vector.append(1)
            else:
                name = member['tags'][0]
                # label_vector.append([1,0])
                label_vector.append(0)
            name_list.append(name)
            class_embedding = class_dict[name]
            x1 = member['points'][0]['x']/width
            y1 = member['points'][0]['y']/height
            x2 = member['points'][1]['x']/width
            y2 = member['points'][1]['y']/height
            center = [x1, y1, x2, y2, (x1+x2)/2,(y1+y2)/2]
            # center_vector.append(center)
            area = [(x2-x1)*(y2-y1)]
            ratio = [(x2-x1)*width/((x2-x1)*width+(y2-y1)*height)]
            node = class_embedding + center + ratio + area
            node_vector.append(node)
        adj_matrix = np.zeros((len(node_vector), len(node_vector)))
        i = 0
        # print(name_list,adj_matrix.shape)
        for mem1 in d['regions']:
            xmin1 = mem1['points'][0]['x']
            ymin1 = mem1['points'][0]['y']
            xmax1 = mem1['points'][1]['x']
            ymax1 = mem1['points'][1]['y']
            j = 0
            for mem2 in d['regions']:
                xmin2 = mem2['points'][0]['x']
                ymin2 = mem2['points'][0]['y']
                xmax2 = mem2['points'][1]['x']
                ymax2 = mem2['points'][1]['y']
                # print(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2)
                dis = bounding_box_overlap(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2)
                adj_matrix[i][j] = dis
                j += 1
            i += 1
        label_np = np.array(label_vector)
        node_np = np.array(node_vector)
        G = Graph(adj_matrix, node_np, label_np, name_list, file_name, primary_key= count)
        Graph_list.append(G)
        count += 1
    return Graph_list, count
