import mindspore as ms
ms.set_context(device_target='GPU')
from mindspore import nn, ops
from mindspore import Tensor
import copy
import torch
import random
import Model
import numpy as np
import math

def train_model(model, Graph_list, n, batch, save_path, wrong_times_path, num_epochs=300, bool_s = False):
    f_dict = open("model_dict.txt", 'wt')
    
    # Set model to train mode
    model.set_train()

    criterion = nn.BCELoss(reduction = 'sum')

    # Define forward function
    def forward_fn(data1, data2, label):
        logits = model(data1, data2)
        scores = logits
        # print(scores, label)
        loss = criterion(scores, label)
        # print(loss)
        return loss, scores

    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.0001)
    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # print("grad_fn", grad_fn)
    # 对第一个输入求梯度

    def train_step(data1, data2, label):
        (loss, scores), grads = grad_fn(data1, data2, label)
        # print("grads", type(grads))
        # for i in range(0, len(grads)):
        #     print("grads", grads[i].shape)
        # print("grads", len(grads), grads[-1])
        optimizer(grads)
        return loss, scores

    
    # Training loop
    wrong_n = -1
    epoch = 0
    wrong_times = [0 for i in range(n)]
    while(wrong_n > 0 or wrong_n < 0):
        if(epoch >= num_epochs):
            break
        acc = 0
        node_num = 0
        wrong_n = 0
        bn = math.ceil(n/batch)
        # random.shuffle(Graph_list)
        for i in range(0, bn):
            loss_v = []
            data_labels_v = []
            preds_v = []
            if (i == bn - 1):
                for j in range(0, n - i*batch):
                    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
                    data_inputs = Tensor(Graph_list[i*batch + j].node, ms.float32)
                    data_labels = Tensor(Graph_list[i*batch + j].label, ms.float32)
                    adj_matrix = Tensor(Graph_list[i*batch + j].adj_matrix, ms.float32)
                    data_inputs = data_inputs.unsqueeze(0)
                    bool_input = copy.deepcopy(data_inputs)
                    bool_input[data_inputs>0] = 1
                    adj_matrix = adj_matrix.unsqueeze(0)
                    data_labels = data_labels.unsqueeze(0)
                    data_labels_v.append(data_labels)

                    ## Step 2: Run the model on the input data
                    print(data_inputs.shape, adj_matrix.shape, data_labels.shape)
                    loss, scores = train_step(data_inputs, adj_matrix, data_labels)
                    # print("loss", loss.asnumpy(), scores)
                    preds_v.append(scores)

                    loss_v.append(loss)
            else:
                for j in range(0, batch):
                    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
                    data_inputs = Tensor(Graph_list[i*batch + j].node, ms.float32)
                    data_labels = Tensor(Graph_list[i*batch + j].label, ms.float32)
                    adj_matrix = Tensor(Graph_list[i*batch + j].adj_matrix, ms.float32)
                    data_inputs = data_inputs.unsqueeze(0)
                    bool_input = copy.deepcopy(data_inputs)
                    bool_input[data_inputs>0] = 1
                    adj_matrix = adj_matrix.unsqueeze(0)
                    data_labels = data_labels.unsqueeze(0)
                    data_labels_v.append(data_labels)

                    ## Step 2: Run the model on the input data
                    loss, scores = train_step(data_inputs, adj_matrix, data_labels)
                    # print("loss", loss.asnumpy(), scores)
                    preds_v.append(scores)

                    loss_v.append(loss)

            loss = loss_v[0]
            for h in range(1,len(loss_v)):
                loss += loss_v[h]
            
            ## Step 6: calculate acc
            for k in range(len(preds_v)):
                # print("ops.abs(preds_v[k]-data_labels_v[k])", ops.abs(preds_v[k]-data_labels_v[k]))
                batch_acc_v = (ops.abs(preds_v[k]-data_labels_v[k])<0.5)

                batch_acc = batch_acc_v.sum().float()
                acc += batch_acc
                if batch_acc != data_labels_v[k].shape[1]:
                    wrong_times[Graph_list[i*batch + k].primary_key] += 1
                    wrong_n += 1
            
                node_num += data_labels_v[k].shape[1]

        print("epoch %2d, wrong_number:%d, Accuracy of the model: %4.2f%%" % (epoch, wrong_n, 100.0*(acc/node_num)))
        
        state_dict = ms.save_checkpoint(optimizer, save_path  + str(epoch) +  '.ckpt')
        if epoch % 10 == 0:
            state_dict = ms.save_checkpoint(optimizer, save_path  + 'all.ckpt')
        epoch += 1
        wrong_times = np.array(wrong_times)
        np.save(wrong_times_path  + str(epoch) +  '.npy',wrong_times)   # 保存为.npy格式

def eval(model, Graph_list, n, batch, save_path, wrong_times_path, num_epochs=300, bool_s = False):
    f_dict = open("model_dict.txt", 'wt')
    
    # Set model to train mode
    model.set_train()

    criterion = nn.BCELoss(reduction = 'sum')

    # Define forward function
    def forward_fn(data1, data2, label):
        logits = model(data1, data2)
        scores = logits
        # print(scores, label)
        loss = criterion(scores, label)
        # print(loss)
        return loss, scores

    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.0001)
    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # print("grad_fn", grad_fn)
    # 对第一个输入求梯度

    def train_step(data1, data2, label):
        (loss, scores), grads = grad_fn(data1, data2, label)
        # print("grads", type(grads))
        # for i in range(0, len(grads)):
        #     print("grads", grads[i].shape)
        # print("grads", len(grads), grads[-1])
        optimizer(grads)
        return loss, scores

    
    # Training loop
    wrong_n = -1
    epoch = 0
    wrong_times = [0 for i in range(n)]
    while(wrong_n > 0 or wrong_n < 0):
        if(epoch >= num_epochs):
            break
        acc = 0
        node_num = 0
        wrong_n = 0
        bn = math.ceil(n/batch)
        # random.shuffle(Graph_list)
        for i in range(0, bn):
            loss_v = []
            data_labels_v = []
            preds_v = []
            if (i == bn - 1):
                for j in range(0, n - i*batch):
                    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
                    data_inputs = Tensor(Graph_list[i*batch + j].node, ms.float32)
                    data_labels = Tensor(Graph_list[i*batch + j].label, ms.float32)
                    adj_matrix = Tensor(Graph_list[i*batch + j].adj_matrix, ms.float32)
                    data_inputs = data_inputs.unsqueeze(0)
                    bool_input = copy.deepcopy(data_inputs)
                    bool_input[data_inputs>0] = 1
                    adj_matrix = adj_matrix.unsqueeze(0)
                    data_labels = data_labels.unsqueeze(0)
                    data_labels_v.append(data_labels)

                    ## Step 2: Run the model on the input data
                    print(data_inputs.shape, adj_matrix.shape, data_labels.shape)
                    loss, scores = train_step(data_inputs, adj_matrix, data_labels)
                    # print("loss", loss.asnumpy(), scores)
                    preds_v.append(scores)

                    loss_v.append(loss)
            else:
                for j in range(0, batch):
                    ## Step 1: Move input data to device (only strictly necessary if we use GPU)
                    data_inputs = Tensor(Graph_list[i*batch + j].node, ms.float32)
                    data_labels = Tensor(Graph_list[i*batch + j].label, ms.float32)
                    adj_matrix = Tensor(Graph_list[i*batch + j].adj_matrix, ms.float32)
                    data_inputs = data_inputs.unsqueeze(0)
                    bool_input = copy.deepcopy(data_inputs)
                    bool_input[data_inputs>0] = 1
                    adj_matrix = adj_matrix.unsqueeze(0)
                    data_labels = data_labels.unsqueeze(0)
                    data_labels_v.append(data_labels)

                    ## Step 2: Run the model on the input data
                    loss, scores = train_step(data_inputs, adj_matrix, data_labels)
                    # print("loss", loss.asnumpy(), scores)
                    preds_v.append(scores)

                    loss_v.append(loss)

            loss = loss_v[0]
            for h in range(1,len(loss_v)):
                loss += loss_v[h]
            
            ## Step 6: calculate acc
            for k in range(len(preds_v)):
                # print("ops.abs(preds_v[k]-data_labels_v[k])", ops.abs(preds_v[k]-data_labels_v[k]))
                batch_acc_v = (ops.abs(preds_v[k]-data_labels_v[k])<0.5)

                batch_acc = batch_acc_v.sum().float()
                acc += batch_acc
                if batch_acc != data_labels_v[k].shape[1]:
                    wrong_times[Graph_list[i*batch + k].primary_key] += 1
                    wrong_n += 1
            
                node_num += data_labels_v[k].shape[1]

        print("epoch %2d, wrong_number:%d, Accuracy of the model: %4.2f%%" % (epoch, wrong_n, 100.0*(acc/node_num)))
