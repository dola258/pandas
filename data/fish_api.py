import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# csv 파일 읽어오기
bream_length = pd.read_csv('C:/Users/Administrator/Desktop/fish/bream_length.csv', header= None)
bream_weight = pd.read_csv('C:/Users/Administrator/Desktop/fish/bream_weight.csv', header= None)
smelt_length = pd.read_csv('C:/Users/Administrator/Desktop/fish/smelt_length.csv', header= None)
smelt_weight = pd.read_csv('C:/Users/Administrator/Desktop/fish/smelt_weight.csv', header= None)

# 읽어온 데이터 모양, 타입, ndarray로 바뀌는지 확인
print(bream_length.shape)
print(type(bream_length))
print(type(bream_length.to_numpy()))


# 시각화하기
plt.scatter(bream_length, bream_weight, c="r")
plt.scatter(smelt_length, smelt_weight, c="b")
plt.ylabel('length')
plt.xlabel('weight')
# plt.show()

# 전부 2차원 => ndarray타입
bream_length = bream_length.to_numpy()
bream_weight = bream_weight.to_numpy()
smelt_length = smelt_length.to_numpy()
smelt_weight = smelt_weight.to_numpy()

#print(bream_length.shape) 

# 1차원으로 변경
bream_length = bream_length.flatten()
bream_weight = bream_weight.flatten()
smelt_length = smelt_length.flatten()
smelt_weight = smelt_weight.flatten()

#print(bream_weight.shape)

# 길이끼리, 무게끼리 합치기 
fish_length = np.hstack((bream_length, smelt_length))
fish_weigth = np.hstack((bream_weight, smelt_weight))

#print(fish_length.shape) # 1차원
#print(fish_weigth.shape) # 1차원

# 피쉬 데이터로 한번에 합치기 - (2, 49) 2차원
fish_data = np.vstack((fish_length, fish_weigth))
#print(fish_data.shape)

# 피쉬 데이터 행 <-> 렬 치환하기 - (49, 2) 2차원
fish_data = np.transpose(fish_data)
#print(fish_data.shape)

# 타겟 데이터 만들기 - 도미 35개, 빙어 14개
bream_target = np.ones(35)
smelt_target = np.zeros(14)

print(bream_target)
print(smelt_target)

# 타겟 데이터 합치기
fish_target = np.hstack((bream_target, smelt_target))
print(fish_target)

# 타겟 데이터 - 2차원 
fish_target = fish_target.reshape(49, -1)
#print(fish_target)

# 데이터 섞기(shuffle)
index = np.arange(49)
np.random.shuffle(index)
print(index)
print(index[:35])

# 테스트 데이터와 훈련 데이터로 구분 (8:2 == 39:10)
train_input = fish_data[index[:39]]
train_target = fish_target[index[:39]]
test_input = fish_data[index[:10]]
test_target = fish_target[index[:10]]

# 훈련데이터 시각화하기
plt.scatter(train_input[:,0], train_input[:,1])
plt.ylabel('weight')
plt.xlabel('length')
#plt.show()

print('fadsfasdfasd')

def getTrains():
    trains = np.hstack((train_target, train_input))
    trains_DataFrame = pd.DataFrame(trains, columns=["train_target", "train_length", "train_weight"])

    return trains_DataFrame
    
def getTestes():
    testes = np.hstack((test_target, test_input))
    testes_DataFrame = pd.DataFrame(testes, columns=["test_target", "test_length", "test_weight"])

    return testes_DataFrame