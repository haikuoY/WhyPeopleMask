import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
# from random import random
import torch
import image_to_graph
import Model
import train


def main():
    ### load data ###

    json_path = './dataset/jsons/'
    xml_path = './dataset/xmls/'


    class_list = image_to_graph.class_list
    class_dict = image_to_graph.load_class_dict(class_list)

    Len = len(class_list)

    d = Len + 8

    
    Graph_list1, count1 = image_to_graph.xml_to_graph(xml_path, class_dict, 0)
    Graph_list2, count2 = image_to_graph.json_to_graph(json_path, class_dict, count1)
    Graph_list = Graph_list1 + Graph_list2
    random.shuffle(Graph_list)

    n = len(Graph_list)
    Graph_list_train = Graph_list[:1000]
    Graph_list_test = Graph_list[1000:]

    # print("total number of data: ", n, "feature dimension: ", d)
    # # return
    # # load model ###
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
    model = Model.NodeClassifer(d, 1).to(device)

    batch = 500
    num_epochs = 100
    save_path = './saved_models/model.pt'
    wrong_times_path = './wrong_times/'
    loss_path = './loss/'

    # train.train_model(model, device, Graph_list, n, batch, save_path, wrong_times_path, num_epochs, True)
    train.train_model2(model, device, Graph_list_train, len(Graph_list_train), batch, save_path, wrong_times_path, loss_path, Graph_list_test, num_epochs, True)
    

if __name__ == "__main__":
    # execute only if run as a script
    main()