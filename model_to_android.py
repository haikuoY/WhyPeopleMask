import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
# from random import random
import torch
import copy
import image_to_graph
import Model
import train
from torch.utils.mobile_optimizer import optimize_for_mobile

def main():
    ## load model ###
    print("model loading start")
    class_list = image_to_graph.class_list
    class_dict = image_to_graph.load_class_dict(class_list)
    Len = len(class_list)
    d = Len + 8
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
    model = Model.NodeClassifer(d, 1).to(device)
    save_path = './saved_models/model.pt'
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print("model loading done")
    
    xml_path = './dataset/xmls/'
    Graph_list, count1 = image_to_graph.xml_to_graph(xml_path, class_dict, 0)
    data_inputs, data_labels, adj_matrix= torch.from_numpy(Graph_list[0].node).to(device).float(), torch.from_numpy(Graph_list[0].label).to(device).float(), torch.from_numpy(Graph_list[0].adj_matrix).to(device).float()
    data_inputs = data_inputs.unsqueeze(0)
    bool_input = copy.deepcopy(data_inputs)
    bool_input[data_inputs>0] = 1
    adj_matrix = adj_matrix.unsqueeze(0)
    data_labels = data_labels.unsqueeze(0)
    data_inputs = torch.autograd.Variable(data_inputs, requires_grad=True)

    # traced_script_module = torch.jit.trace(model,[data_inputs, adj_matrix])
    traced_script_module = torch.jit.script(model,[data_inputs, adj_matrix])
    traced_script_module_optimied = optimize_for_mobile(traced_script_module)
    traced_script_module_optimied._save_for_lite_interpreter(r"/datapool/workspace/yuhaikuo/IPDH/saved_models/android/0503_190_2.ptl")
     
if __name__ == "__main__":
    # execute only if run as a script
    main()