# -*- coding: utf-8 -*-

import numpy as np
# from sets import Set
import pickle

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from utils import *
from tflearn.activations import relu
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
from optparse import OptionParser
import torch
import deepdish as dd

saver = tf.train.Saver()

# 输入参数
# 解析命令行选项 嵌入维度 ； 待截取的所有梯度的平方和 ；项目矩阵的维度 ； 测试场景 ； 正负比率
parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d")
parser.add_option("-n", "--n", default=1, help="global norm to be clipped")
parser.add_option("-k", "--k", default=512, help="The dimension of project matrices k")
parser.add_option("-t", "--t", default="o", help="Test scenario")
parser.add_option("-r", "--r", default="ten", help="positive negative ratio")

(opts, args) = parser.parse_args()  # 其中'values'是values实例(包含所有选项值)，'args'是解析选项后剩下的参数列表


def check_symmetric(a, tol=1e-8):
    # 检查对称
    return np.allclose(a, a.T, atol=tol)


def row_normalize(a_matrix, substract_self_loop):
    # 行规格化
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


# load network 加载网络以及标准化
network_path = '../data/'

drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
# print('loaded drug drug', check_symmetric(drug_drug), np.shape(drug_drug))
true_drug = 708  # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database(化合物检索自ZINC15数据库)
drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
drug_chemical = drug_chemical[:true_drug, :true_drug]
# print 'loaded drug chemical', check_symmetric(drug_chemical), np.shape(drug_chemical)
drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
# print 'loaded drug disease', np.shape(drug_disease)
drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')
# print 'loaded drug sideffect', np.shape(drug_sideeffect)
disease_drug = drug_disease.T
sideeffect_drug = drug_sideeffect.T

protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
# print 'loaded protein protein', check_symmetric(protein_protein), np.shape(protein_protein)
protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt')
# print 'loaded protein sequence', check_symmetric(protein_sequence), np.shape(protein_sequence)
protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')
# print 'loaded protein disease', np.shape(protein_disease)
disease_protein = protein_disease.T

# normalize network for mean pooling aggregation 规格化网络的平均池聚合
drug_drug_normalize = row_normalize(drug_drug, True)
drug_chemical_normalize = row_normalize(drug_chemical, True)
drug_disease_normalize = row_normalize(drug_disease, False)
drug_sideeffect_normalize = row_normalize(drug_sideeffect, False)

protein_protein_normalize = row_normalize(protein_protein, True)
protein_sequence_normalize = row_normalize(protein_sequence, True)
protein_disease_normalize = row_normalize(protein_disease, False)

disease_drug_normalize = row_normalize(disease_drug, False)
disease_protein_normalize = row_normalize(disease_protein, False)
sideeffect_drug_normalize = row_normalize(sideeffect_drug, False)

# define computation graph 定义计算图
num_drug = len(drug_drug_normalize)
num_protein = len(protein_protein_normalize)
num_disease = len(disease_protein_normalize)
num_sideeffect = len(sideeffect_drug_normalize)

dim_drug = int(opts.d)
dim_protein = int(opts.d)
dim_disease = int(opts.d)
dim_sideeffect = int(opts.d)
dim_pred = int(opts.k)  # 项目矩阵维度
dim_pass = int(opts.d)


# 定义模型
class Model(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        # inputs 构建模型，输入属性
        # placeholder()作用：为一个张量插入一个占位符，该张量将始终被填充。
        self.drug_drug = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_drug_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_chemical = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_chemical_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_disease = tf.placeholder(tf.float32, [num_drug, num_disease])
        self.drug_disease_normalize = tf.placeholder(tf.float32, [num_drug, num_disease])

        self.drug_sideeffect = tf.placeholder(tf.float32, [num_drug, num_sideeffect])
        self.drug_sideeffect_normalize = tf.placeholder(tf.float32, [num_drug, num_sideeffect])

        self.protein_protein = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_protein_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.protein_sequence = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_sequence_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.protein_disease = tf.placeholder(tf.float32, [num_protein, num_disease])
        self.protein_disease_normalize = tf.placeholder(tf.float32, [num_protein, num_disease])

        self.disease_drug = tf.placeholder(tf.float32, [num_disease, num_drug])
        self.disease_drug_normalize = tf.placeholder(tf.float32, [num_disease, num_drug])

        self.disease_protein = tf.placeholder(tf.float32, [num_disease, num_protein])
        self.disease_protein_normalize = tf.placeholder(tf.float32, [num_disease, num_protein])

        self.sideeffect_drug = tf.placeholder(tf.float32, [num_sideeffect, num_drug])
        self.sideeffect_drug_normalize = tf.placeholder(tf.float32, [num_sideeffect, num_drug])

        self.drug_protein = tf.placeholder(tf.float32, [num_drug, num_protein])
        self.drug_protein_normalize = tf.placeholder(tf.float32, [num_drug, num_protein])

        self.protein_drug = tf.placeholder(tf.float32, [num_protein, num_drug])
        self.protein_drug_normalize = tf.placeholder(tf.float32, [num_protein, num_drug])

        self.drug_protein_mask = tf.placeholder(tf.float32, [num_drug, num_protein])

        # features 特性 embedding 嵌入层
        self.drug_embedding = weight_variable([num_drug, dim_drug])
        self.protein_embedding = weight_variable([num_protein, dim_protein])
        self.disease_embedding = weight_variable([num_disease, dim_disease])
        self.sideeffect_embedding = weight_variable([num_sideeffect, dim_sideeffect])

        # 包装
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.drug_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.protein_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.disease_embedding))
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.sideeffect_embedding))

        # feature passing weights (maybe different types of nodes can use different weights) 特性传递权值(可能不同类型的节点可以使用不同的权值)
        W0 = weight_variable([dim_pass + dim_drug, dim_drug])
        b0 = bias_variable([dim_drug])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))

        # passing 1 times (can be easily extended to multiple passes) 传递1次(可以很容易扩展到多次传递)
        # 药向量
        drug_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.drug_drug_normalize, a_layer(self.drug_embedding, dim_pass)) + \
                       tf.matmul(self.drug_chemical_normalize, a_layer(self.drug_embedding, dim_pass)) + \
                       tf.matmul(self.drug_disease_normalize, a_layer(self.disease_embedding, dim_pass)) + \
                       tf.matmul(self.drug_sideeffect_normalize, a_layer(self.sideeffect_embedding, dim_pass)) + \
                       tf.matmul(self.drug_protein_normalize, a_layer(self.protein_embedding, dim_pass)), \
                       self.drug_embedding], axis=1), W0) + b0), dim=1)

        # 蛋白质向量
        protein_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.protein_protein_normalize, a_layer(self.protein_embedding, dim_pass)) + \
                       tf.matmul(self.protein_sequence_normalize, a_layer(self.protein_embedding, dim_pass)) + \
                       tf.matmul(self.protein_disease_normalize, a_layer(self.disease_embedding, dim_pass)) + \
                       tf.matmul(self.protein_drug_normalize, a_layer(self.drug_embedding, dim_pass)), \
                       self.protein_embedding], axis=1), W0) + b0), dim=1)

        # 疾病向量
        disease_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.disease_drug_normalize, a_layer(self.drug_embedding, dim_pass)) + \
                       tf.matmul(self.disease_protein_normalize, a_layer(self.protein_embedding, dim_pass)), \
                       self.disease_embedding], axis=1), W0) + b0), dim=1)

        # 副作用向量
        sideeffect_vector1 = tf.nn.l2_normalize(relu(tf.matmul(
            tf.concat([tf.matmul(self.sideeffect_drug_normalize, a_layer(self.drug_embedding, dim_pass)), \
                       self.sideeffect_embedding], axis=1), W0) + b0), dim=1)

        # 将建立的向量输入到模型当中
        self.drug_representation = drug_vector1
        self.protein_representation = protein_vector1
        self.disease_representation = disease_vector1
        self.sideeffect_representation = sideeffect_vector1

        # reconstructing networks 重构网络
        self.drug_drug_reconstruct = bi_layer(self.drug_representation, self.drug_representation, sym=True,
                                              dim_pred=dim_pred)
        self.drug_drug_reconstruct_loss = tf.reduce_sum(
            tf.multiply((self.drug_drug_reconstruct - self.drug_drug), (self.drug_drug_reconstruct - self.drug_drug)))

        self.drug_chemical_reconstruct = bi_layer(self.drug_representation, self.drug_representation, sym=True,
                                                  dim_pred=dim_pred)
        self.drug_chemical_reconstruct_loss = tf.reduce_sum(
            tf.multiply((self.drug_chemical_reconstruct - self.drug_chemical),
                        (self.drug_chemical_reconstruct - self.drug_chemical)))

        self.drug_disease_reconstruct = bi_layer(self.drug_representation, self.disease_representation, sym=False,
                                                 dim_pred=dim_pred)
        self.drug_disease_reconstruct_loss = tf.reduce_sum(
            tf.multiply((self.drug_disease_reconstruct - self.drug_disease),
                        (self.drug_disease_reconstruct - self.drug_disease)))

        self.drug_sideeffect_reconstruct = bi_layer(self.drug_representation, self.sideeffect_representation, sym=False,
                                                    dim_pred=dim_pred)
        self.drug_sideeffect_reconstruct_loss = tf.reduce_sum(
            tf.multiply((self.drug_sideeffect_reconstruct - self.drug_sideeffect),
                        (self.drug_sideeffect_reconstruct - self.drug_sideeffect)))

        self.protein_protein_reconstruct = bi_layer(self.protein_representation, self.protein_representation, sym=True,
                                                    dim_pred=dim_pred)
        self.protein_protein_reconstruct_loss = tf.reduce_sum(
            tf.multiply((self.protein_protein_reconstruct - self.protein_protein),
                        (self.protein_protein_reconstruct - self.protein_protein)))

        self.protein_sequence_reconstruct = bi_layer(self.protein_representation, self.protein_representation, sym=True,
                                                     dim_pred=dim_pred)
        self.protein_sequence_reconstruct_loss = tf.reduce_sum(
            tf.multiply((self.protein_sequence_reconstruct - self.protein_sequence),
                        (self.protein_sequence_reconstruct - self.protein_sequence)))

        self.protein_disease_reconstruct = bi_layer(self.protein_representation, self.disease_representation, sym=False,
                                                    dim_pred=dim_pred)
        self.protein_disease_reconstruct_loss = tf.reduce_sum(
            tf.multiply((self.protein_disease_reconstruct - self.protein_disease),
                        (self.protein_disease_reconstruct - self.protein_disease)))

        self.drug_protein_reconstruct = bi_layer(self.drug_representation, self.protein_representation, sym=False,
                                                 dim_pred=dim_pred)
        tmp = tf.multiply(self.drug_protein_mask, (self.drug_protein_reconstruct - self.drug_protein))
        self.drug_protein_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp))

        self.l2_loss = tf.add_n(tf.get_collection("l2_reg"))

        # 计算丢失
        self.loss = self.drug_protein_reconstruct_loss + 1.0 * (
                    self.drug_drug_reconstruct_loss + self.drug_chemical_reconstruct_loss +
                    self.drug_disease_reconstruct_loss + self.drug_sideeffect_reconstruct_loss +
                    self.protein_protein_reconstruct_loss + self.protein_sequence_reconstruct_loss +
                    self.protein_disease_reconstruct_loss) + self.l2_loss


# 将graph赋为默认图
graph = tf.get_default_graph()
# graph.as_default()
# 返回一个上下文管理器，该管理器使此图成为默认图。
# 如果您想在同一过程中创建多个图，则应该使用此方法。 为了方便起见，提供了一个全局默认图，如果您没有显式地创建一个新图，那么所有的操作都将添加到这个图中。 使用这个方法和with关键字来指定在块范围内创建的操作应该添加到这个图中。
# 默认图形是当前线程的一个属性。 如果你创建了一个新线程，并且希望在该线程中使用默认的图形，你必须显式地在该线程的函数中添加一个with g.a as_default():。
# 下面的代码示例是等价的:
# ' ' ' python # 1。 使用Graph.as_default(): g = tf.Graph() with g.as_default():
# C = tf.constant(5.0) assert C .graph is g
# # 2。 使用tf.Graph().as_default()构造和创建default:
# C = tf.constant(5.0) assert C .graph is g
# ｀｀｀
# 返回:
# 使用此图作为默认图的上下文管理器。
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    dti_loss = model.drug_protein_reconstruct_loss

    # 优化
    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, int(opts.n))
    optimizer = optimize.apply_gradients(zip(gradients, variables))

    eval_pred = model.drug_protein_reconstruct


# 训练与评估
def train_and_evaluate(DTItrain, DTIvalid, DTItest, graph, verbose=True, num_steps=4000):
    drug_protein = np.zeros((num_drug, num_protein))
    mask = np.zeros((num_drug, num_protein))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein, False)
    protein_drug_normalize = row_normalize(protein_drug, False)

    lr = 0.001

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for i in range(num_steps):
            _, tloss, dtiloss, results = sess.run([optimizer, total_loss, dti_loss, eval_pred],
                                                  feed_dict={model.drug_drug: drug_drug,
                                                             model.drug_drug_normalize: drug_drug_normalize,
                                                             model.drug_chemical: drug_chemical,
                                                             model.drug_chemical_normalize: drug_chemical_normalize,
                                                             model.drug_disease: drug_disease,
                                                             model.drug_disease_normalize: drug_disease_normalize,
                                                             model.drug_sideeffect: drug_sideeffect,
                                                             model.drug_sideeffect_normalize: drug_sideeffect_normalize,
                                                             model.protein_protein: protein_protein,
                                                             model.protein_protein_normalize: protein_protein_normalize,
                                                             model.protein_sequence: protein_sequence,
                                                             model.protein_sequence_normalize: protein_sequence_normalize,
                                                             model.protein_disease: protein_disease,
                                                             model.protein_disease_normalize: protein_disease_normalize,
                                                             model.disease_drug: disease_drug,
                                                             model.disease_drug_normalize: disease_drug_normalize,
                                                             model.disease_protein: disease_protein,
                                                             model.disease_protein_normalize: disease_protein_normalize,
                                                             model.sideeffect_drug: sideeffect_drug,
                                                             model.sideeffect_drug_normalize: sideeffect_drug_normalize,
                                                             model.drug_protein: drug_protein,
                                                             model.drug_protein_normalize: drug_protein_normalize,
                                                             model.protein_drug: protein_drug,
                                                             model.protein_drug_normalize: protein_drug_normalize,
                                                             model.drug_protein_mask: mask,
                                                             learning_rate: lr})
            # every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
            # 每25步梯度下降，评估性能，其他选择这个数字是可能的
            if i % 25 == 0 and verbose == True:
                print('step', i, 'total and dtiloss', tloss, dtiloss)

                pred_list = []
                ground_truth = []
                for ele in DTIvalid:
                    pred_list.append(results[ele[0], ele[1]])
                    ground_truth.append(ele[2])
                valid_auc = roc_auc_score(ground_truth, pred_list)
                valid_aupr = average_precision_score(ground_truth, pred_list)
                if valid_aupr >= best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    best_valid_auc = valid_auc
                    pred_list = []
                    ground_truth = []
                    for ele in DTItest:
                        pred_list.append(results[ele[0], ele[1]])
                        ground_truth.append(ele[2])
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)
                print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
                saver.save(sess, 'my_test_model')
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr

# auc -> ROC曲线下的面积
# aupc -> pr曲线下的面积，PR即召回率和正确率组成的曲线图
test_auc_round = []
test_aupr_round = []

for r in range(10):
    # 10轮训练与评估
    print('sample round', r + 1)
    if opts.t == 'o':
        dti_o = np.loadtxt(network_path + 'mat_drug_protein.txt')
    else:
        dti_o = np.loadtxt(network_path + 'mat_drug_protein_' + opts.t + '.txt')

    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_o[i][j]) == 0:
                whole_negative_index.append([i, j])

    if opts.r == 'ten':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=10 * len(whole_positive_index), replace=False)
    elif opts.r == 'all':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)), size=len(whole_negative_index),
                                                 replace=False)
    else:
        print('wrong positive negative ratio')
        break

    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    if opts.t == 'unique':
        whole_positive_index_test = []
        whole_negative_index_test = []
        for i in range(np.shape(dti_o)[0]):
            for j in range(np.shape(dti_o)[1]):
                if int(dti_o[i][j]) == 3:
                    whole_positive_index_test.append([i, j])
                elif int(dti_o[i][j]) == 2:
                    whole_negative_index_test.append([i, j])

        if opts.r == 'ten':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),
                                                          size=10 * len(whole_positive_index_test), replace=False)
        elif opts.r == 'all':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),
                                                          size=whole_negative_index_test, replace=False)
        else:
            print('wrong positive negative ratio')
            break
        data_set_test = np.zeros((len(negative_sample_index_test) + len(whole_positive_index_test), 3), dtype=int)
        count = 0
        for i in whole_positive_index_test:
            data_set_test[count][0] = i[0]
            data_set_test[count][1] = i[1]
            data_set_test[count][2] = 1
            count += 1
        for i in negative_sample_index_test:
            data_set_test[count][0] = whole_negative_index_test[i][0]
            data_set_test[count][1] = whole_negative_index_test[i][1]
            data_set_test[count][2] = 0
            count += 1

        DTItrain = data_set
        DTItest = data_set_test
        rs = np.random.randint(0, 1000, 1)[0]
        DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05, random_state=rs)
        v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest,
                                                          graph=graph, num_steps=3000)

        # 将值添加到 auc 与 aupr中
        test_auc_round.append(t_auc)
        test_aupr_round.append(t_aupr)
        np.savetxt('test_auc', test_auc_round)
        np.savetxt('test_aupr', test_aupr_round)

    else:
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0, 1000, 1)[0]
        # StratifiedKFold()
        # 分层K-Folds cross-validator。
        # 提供训练/测试指标来分割训练/测试集中的数据。
        # 这个交叉验证对象是KFold的变体，它返回分层的折叠。 折叠是通过保留每个类的样本百分比来实现的。
        # 更多信息请参阅用户指南。
        # 该实现旨在:
        # 生成测试集，以便所有测试集都包含相同的类分布，或者尽可能接近。
        # 对类标签保持不变:将y = ["Happy"， "Sad"]重新标记为y =[1,0]不应改变生成的索引。
        # 保留数据集排序中的顺序依赖，当shuffle=False时:在某个测试集中，来自类k的所有样本在y中是连续的，或者在y中被来自非类k的样本分隔。
        # 生成测试集，其中最小值和最大值最多相差一个样本。
        # kf = StratifiedKFold(data_set[:,2], n_folds=10, shuffle=True, random_state=rs)
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs)
        # print(kf.get_n_splits(data_set[:,2]))

        # for train_index, test_index in kf:
        for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05, random_state=rs)

            v_auc, v_aupr, t_auc, t_aupr = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest,
                                                              graph=graph, num_steps=3000)
            # 将 auc 与 aupr 的值保存到临时列表中，以便之后求取平均值
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)

        # 将值添加到 auc 与 aupr中 mean() -> 返回数组元素的平均值
        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        np.savetxt('test_auc', test_auc_round)
        np.savetxt('test_aupr', test_aupr_round)
