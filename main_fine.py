import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import StandardScaler  # 归一化
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV  # 在sklearn中主要是使用GridSearchCV调参
import numpy as np
import xlrd
import pickle

#数据显示和预处理
if __name__ == '__main__':

 ##导入数据集
    def matrix(path):
        table = xlrd.open_workbook(path).sheets()[0]  # 获取第一个sheet表
        row = table.nrows  # 行数
        col = table.ncols  # 列数
        datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
        for x in range(col):
            try:
                cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
                datamatrix[:, x] = cols  # 按列把数据存进矩阵中
            except:
                print(x)

        # print(datamatrix.shape)
        return datamatrix


    path = r"E:\python\data\data_y_sss.xlsx"
    datamatrix = matrix(path)
    y_dmrs=datamatrix
    path = r"E:\python\data\data_signal_sss.xlsx"
    datamatrix = matrix(path)
    x_dmrs = datamatrix
    print(y_dmrs.shape)
    print(x_dmrs.shape)
    x=x_dmrs

    y=y_dmrs
    y = y.ravel()

## 外部读取数据集
    #path = r"E:\python\data\test_dmrsRex.xlsx"
    #datamatrix = matrix(path)
    #x_dmrs_test = datamatrix
    #print('datared')
    #print(datamatrix.shape)
    print("x:", x.shape)
    print("y:", y.shape)

#*******************************

##数据集分类为训练集和测试集
    num=x.shape[0]
    ratio=7/3
    num_test=int(num/(1+ratio)) #测试样本数目
    num_train=num-num_test #训练集样本数目
    index=np.arange(num)
    np.random.shuffle(index)
    x_test=x[index[:num_test], :] #取出洗牌后前num_test 作为测试集
    y_test=y[index[:num_test]]
    x_train=x[index[num_test:], :] #剩余作为训练集
    y_train=y[index[num_test:]]

#************************************

## 归一化过程
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    #
    #
    pickle.dump(scaler, open("sca_1.dat", "wb"))  # open("dtr.dat","wb")意思是打开叫"dtr.dat"的文件,操作方式是写入二进制数据
    x_test = scaler.transform(x_test)    #数据集训练

# ****************************

 ##SVC默认设置—— C=1; gamma=1/(特征){dmrs=1/288};cache_size=200;max_iter{最大迭代次数}；decision_function_shape=“ovo”或者“ovr”,多分类参数
 ##random_state{数据洗牌时的种子数}；
    #clf_linear=svm.SVC(decision_function_shape="ovo",kernel="linear")
    clf_rbf = svm.SVC(decision_function_shape="ovo", kernel="rbf", C=5, gamma=0.0005, probability=True)
    #clf_linear.fit(x_train,y_train)
    clf_rbf.fit(x_train, y_train)
    print("%%%%")
    kk = clf_rbf.predict_proba(x_train)
    print("%%%%")

    x_test_b = clf_rbf.predict_proba(x_test)
    print("%%%%", x_test_b)
    # kk = clf_rbf.decision_function(x_train)
    pickle.dump(clf_rbf, open("dtr_1.dat", "wb"))  # open("dtr.dat","wb")意思是打开叫"dtr.dat"的文件,操作方式是写入二进制数据
    clf_rbf_b = svm.SVC(decision_function_shape="ovo", kernel="linear")
    scaler_b = MinMaxScaler()
    kk = scaler_b.fit_transform(kk)
    x_test_b = scaler_b.transform(x_test_b)
    pickle.dump(scaler_b, open("sca_2.dat", "wb"))  # open("dtr.dat","wb")意思是打开叫"dtr.dat"的文件,操作方式是写入二进制数据
    clf_rbf_b.fit(kk, y_train)
    print("%%%%")
    # print("score:", kk.shape)
    print("数组的形状：", x_train.shape)
    print("数组的形状：", x_test.shape)
    y_test_pre_rbf = clf_rbf.predict(x_test)
    y_test_pre_rbf_b = clf_rbf_b.predict(x_test_b)
    ## 读出模型
    pickle.dump(clf_rbf_b, open("dtr_2.dat", "wb"))  # open("dtr.dat","wb")意思是打开叫"dtr.dat"的文件,操作方式是写入二进制数据
    # loaded_model = pickle.load(open("dtr.dat", "rb"))

    #********************************

    # y_test_pre_rbf = loaded_model.predict(x_test)
      #y_test_rbf = loaded_model.predict(x_dmrs_test)
    #print('y_test_rbf:', y_test_rbf)
    #判断训练效果
    # print('y_test:', y_test)
    # print('y_test_pre_rbf', y_test_pre_rbf)

    #acc_linear=sum(y_test_pre_linear == y_test)/num_test
    print('num_test:', num_test)
   # print('linear kernel:The accuracy is', acc_linear)
    #   print("3:", y_test)
    acc_rbf=sum(y_test_pre_rbf == y_test)/num_test
    print('rbf kernel:The accuracy is', acc_rbf)

    acc_rbf_b = sum(y_test_pre_rbf_b == y_test)/num_test
    print('two rbf kernel:The accuracy is', acc_rbf_b)