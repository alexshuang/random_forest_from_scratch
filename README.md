![](https://upload-images.jianshu.io/upload_images/13575947-20c72e258048e83a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 机器自己摸索决策逻辑

在[【Kaggle】房价预测模型在房产投资场景的应用](https://www.jianshu.com/p/533ac1002cdb)一文中，我提到随机森林（Random Forest）算法模型具有良好的数据解释性，本文就用从零开始写该算法的方式，希望能彻底讲清楚随机森林的工作原理。

模型的另一种表达方式是函数：$y = func(x)$。所谓的数据解释性，指的就是人能理解$func()$是如何通过x生成y的。换句话说，人能理解$func()$的内容。比如$y = ax + b$，统计学最爱的逻辑回归直线，它只有$a、b$这2个变量，$a$就是这条直线的斜率，$b$是直线的截距，通过调整直线斜率和截距，就可以把两类事物区分开来，或是预测y的区域。

![](https://upload-images.jianshu.io/upload_images/13575947-1ffe0df0ea2f072d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

随机森林的基础是决策树，从结构上看，它是决策树的集合。决策树是一种基于因果关系，通过逻辑论证推理来学习知识的模型。之所以说人能理解随机森林，就是因为决策树的方法论也是人理解和探索世界的方法论，只是现在人把这些工作交给机器来做了而已。

## 随机和森林

随机森林模型的工作原理并不复杂，它的核心思想都写在它的脑门上了，就是“随机”和“森林”。
，为数据和决策增加更多随机概率。

- “森林”指的是由N棵决策树集成（Ensembling）在一起，这样可以降低单棵决策树的预测方差（Variance）。
- “随机”指的是为模型增加更多随机性，主要体现在以下两方面：
   - 数据采样的随机性。随机森林采用自举法（bootstrapping）的重采样方法，让每颗决策树的采样数据尽可能不同。
   - 决策的随机性。决策树的工作就是做决策，所谓的为决策增加随机性，指的是增加“差决策”被选择的概率。在做选择时，有好不选而选差的，这听起来很荒谬，但这就好像是小说《天龙八部》里虚竹破珍珑棋局。虚竹恰恰是自杀式地下了一手任何会下棋的人都不会考虑的错棋之后，才迎来真正的好棋，而那些一直在理所当然地下“好棋”的高手都掉进了珍珑棋局的陷阱里，棋路越下越窄，最后被逼进绝路。

---

## Let's get to work

对随机森林的工作原理有了初步了解后，让我们开始写代码吧。数据集还是使用[【Kaggle】房价预测模型在房产投资场景的应用](https://www.jianshu.com/p/533ac1002cdb)的数据，为了验证模型的准确性，用sklearn的随机森林模型（RandomForestRegressor）作为对标模型。

首先，为了简化模型，我只用"OverallQual"和"GrLivArea"这两列数据，也就是说，模型现在只需要处理2个特征。

```
x = train_df[['OverallQual', 'GrLivArea']].copy()
trn_x, val_x, trn_y, val_y = train_test_split(x, y, test_size=0.3)
n = 1000
trn_x, trn_y = trn_x[:n], trn_y[:n]
```

### 随机森林模型

```
class RandomForest():
  def __init__(self, x, y, nr_trees, sample_sz=1000):
    self.x, self.y, self.nr_trees, self.sample_sz = x, y, nr_trees, sample_sz
    self.trees = [self.create_tree() for _ in range(nr_trees)]
  
  def create_tree(self):
    idxs = np.random.permutation(len(self.x))[:self.sample_sz]
    return DecisionTree(self.x.iloc[idxs], self.y[idxs], min_leaf=5)
```

可以看到，随机森林模型参数有两个：nr_trees（决策树数）和sample_sz（样本采样数）。前文提到，决策树会采用重采样（bootstrapping）来保证采样的随机性，然而我们的模型并没有采用重采样，而是采用随机不重复采样：**"idxs = np.random.permutation(len(self.x))[:self.sample_sz]"**。

之所以不使用重采样，是因为随机森林是20多年前发明的，重采样只是当时小数据时代的不得已之举，现在是大数据时代，已经不存在数据不足的问题，随机不重复采样才是更好之选。

这里将采样数设置为1000（**sample_sz=1000**），是为了保证我们的模型和对标模型没有数据集的差异（训练集大小就是1000）。

### 决策树模型

决策树是一个根据事物属性对事物分类的树形结构。比如病人小李去医院看病，医生会先问“哪里不舒服啦”，小李说“体乏、流鼻涕、咳嗽”，这时医生会让小李先去量体温并做血常规检测，因为他判断小李很可能得了感冒，但也不排除是流感甚至肺炎的可能。如何判断病人具体得到那种病呢，医生这时候就会用到一个决策树：体温低于38.2°，血常规正常，那就是一般感冒；体温高于38.2°，病情超过1周了，那可能是流感，加上血常规某项指标低于标准值，那还需要让小李再去做另一项肺部检查才能排除他是否得了肺炎。

![](https://upload-images.jianshu.io/upload_images/13575947-cffca8d148dd2baf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

医生之所以能根据体温和检查报告来判断病人得了什么病，是因为他学了几年的医学知识，从已知的病理出发：感冒、流感和肺炎都会产生小李描述的症状，因此先假设小李得了这三种病，接着收集数据来验证假设。

决策树则是采用逆向学习的方法，从病例的已知结果出发逆向找出因果链条。虽然两者的学习方法不同，但结果是相同的，都是找出因果关系。

```
class DecisionTree():
  def __init__(self, x, y, idxs=None, min_leaf=1):
    if idxs is None: idxs = np.arange(len(x))
    self.x, self.y, self.min_leaf, self.idxs = x.reset_index(drop=True), y, min_leaf, idxs
    self.n, self.c = len(idxs), x.shape[1]
    self.val = self.y[idxs].mean()
    self.score = float('inf')
    self.find_split()
  
  def find_split(self):
    for c in range(self.c):
      self.find_better_split(c)
  
  def find_better_split(self, var_idx):
    x, y = self.x.iloc[self.idxs, var_idx].values, self.y[self.idxs]
    for i in range(self.n - self.min_leaf):
      l_idxs = (x <= x[i])
      r_idxs = x > x[i]
      if r_idxs.sum() == 0: continue
      l_std = y[l_idxs].std()
      r_std = y[r_idxs].std()
      score = l_std * l_idxs.sum() + r_std * r_idxs.sum()
      if score < self.score:
        self.score, self.var_idx, self.split_val = score, var_idx, x[i]
        
  @property
  def split_var(self): return self.x.columns[self.var_idx]
  
  @property
  def is_leaf(self): return self.score == float('inf')
  
  def __repr__(self):
    txt = f'samples: {self.n}, value: {self.val}, score: {self.score}, '
    if not self.is_leaf:
      txt += f'split variable: {self.split_var}, split value: {self.split_val}'
    return txt
```

find_better_split()是决策树进行决策的函数，**var_idx**是column id，首先需要拿出所有样本的同一列作为x，以小李看病例子为例，x表示1000个病人的体温（或血常规检查报告），y表示这1000个病人对应的病症。

决策树是逆向寻找因果链条的，然而树中的链条有很多，假设链条上有3个节点，每个节点有True和False两个分叉，能形成$2^3$条链，机器是如何判断哪条链才是正确的因果关系链呢？

其实机器并没有更好的办法，它只能一个个试。比如它先以样本1的体温为参照点，把所有体温小于等于样本1的样本归为一类，把所有体温大于样本2的样本归为另一类。因为已知样本的真实分类，因此可以统计这种分类方法的准确率水平，用变量**score**表示。用同样的方法，再拿样本2、样本3......样本1000作为参照点，最终选出score值最小的那个样本，它的体温就是最准确的分类特征，例如体温小于等于$38.2^\circ$就是普通感冒，大于这个温度就是流感或肺炎。

**score**称为“混杂度”，它表示样本里分类的纯度，score越小表示分类越准。因为你不可能只经过一次分类就能把样本都分完整了，因此，每次分类，或者说决策的过程，就是减少score值的过程。可以把计算score的函数理解为损失函数。

随机森林提供了很多计算score的函数，如mse（默认值）、cross entropy、gini。MSE的平方根就是RMSE（root mean square error），它指的是样本观察值与均值的平均距离，值越小表示模型效果越好。RMSE的另一种近似的表达是标准差，它也是观察值与均值的距离，值越小表示模型越好。相比RMSE，标准差的计算量要小，因此这里用标准差来计算score。

**min_leaf**可以用来调控参与决策的样本数，该值越大，表示参与决策的样本数越少，决策数也会跟着减少。在这里它的默认是5，但你不要以为只是少了5个样本而已。因为决策树需要做大量的决策（分叉），如果每次决策都少5个样本，那结果会少许多决策。min_leaf=2，每颗决策树的决策数是$log_2^{1000} - 1$，min_leaf每增加一倍，决策数差不多就会减少一半。

既然决策树的工作就是不断决策，那么这个决策循环什么时候结束呢？当无法再对现在的样本分类的时候，就是决策树的终点，它们也称为叶节点，可以用**is_leaf**来查询是否是叶节点。叶子节点的score == inf，即无限大，表示它们不参与决策。

## 单次决策

```
rf = RandomForest(trn_x, trn_y, 1)
print(rf.trees[0])

samples: 1000, value: 12.030368441695023, score: 288.68995409377146,
split variable: OverallQual, split value: 6
```
![Figure 1](https://upload-images.jianshu.io/upload_images/13575947-3b0eaf69faf4cc26.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

前面用了很长的篇幅来解释决策树的工作原理，接了来就是验证决策树模型的表现了。Figure1是对标模型的决策树图，可以看到，两个模型都是用OverallQual这个特征来分类，区别在于"<= 6.5"和"<=6"。很奇怪，明明OverallQual都是整数，6.5是从何而来的？我猜测"6.5 = 6 + 误差修正"，它这样做的目的是希望让分类更精确。

```
%timeit RandomForest(trn_x, trn_y, 1)
10 loops, best of 3: 133 ms per loop

%timeit m.fit(trn_x, trn_y)
1000 loops, best of 3: 1.32 ms per loop
```
%timeit是用来测试程序运行所花费的时间的，相比%time，它取多次运行的均值，因此更精确。可以看到，现在这个决策树运行一次需要133ms，比对标模型慢100倍。然后我用%prun来查看每个模块具体的时间花销，不出所料，问题在find_better_split()。

**x <= x[i]**以及**x > x[i]**的底层代码也是用for循环实现的，那么find_better_split()的时间复杂度就是$O(n^2)$。显然，如果x是已经升序排序好的，那**x <= x[i]**就可以少做很多比较，而快速排序的时间复杂度是$O(nlog(n))$，远小于$O(n^2)$。

![](https://upload-images.jianshu.io/upload_images/13575947-ac978a126f005857.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

除此之外，np.std()也比我想象中更花时间（14ms），因此，可以换一种计算方法来提升速度。Figure 2是标准的标准差计算公式，Figure 3是我们要采用的新公式。新公式不需要计算每个观察值和均值的差，计算量更少，只需要求$X^2$和$X$的均值（E[X] == 均值）即可。

![Figure 2](https://upload-images.jianshu.io/upload_images/13575947-3c336cd3637f59a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Figure 3](https://upload-images.jianshu.io/upload_images/13575947-838c9ade78db2b29.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
def calc_std(n, s, s2): return np.sqrt(s2/n - (s/n)**2)

def find_better_split(self, var_idx):
  x, y = self.x.iloc[self.idxs, var_idx].values, self.y[self.idxs]
  idxs = np.argsort(x)
  x_sorted, y_sorted = x[idxs], y[idxs]
  r_cnt, r_s, r_s2 = len(x_sorted), y_sorted.sum(), (y_sorted**2).sum()
  l_cnt, l_s, l_s2 = 0., 0., 0.

  for i in range(self.n - self.min_leaf):
    l_cnt += 1; r_cnt -= 1
    l_s += y_sorted[i]; r_s -= y_sorted[i]
    l_s2 += y_sorted[i]**2; r_s2 -= y_sorted[i]**2
    if x_sorted[i] == x_sorted[i + 1]: continue

    l_std = calc_std(l_cnt, l_s, l_s2)
    r_std = calc_std(r_cnt, r_s, r_s2)
    score = l_cnt * l_std + r_cnt * r_std
    if score < self.score:
      self.score, self.var_idx, self.split_val = score, var_idx, x_sorted[i]
```
可以看到，排序后的x，不会重复用同一类的样本做参考点重复决策，少做了很多无用功，因此运行一次程序只需要8ms。

```
rf = RandomForest(trn_x, trn_y, 1)
print(rf.trees[0])
%timeit RandomForest(trn_x, trn_y, 1)

samples: 1000, value: 12.030368441695023, score: 288.6899540938478, split variable: OverallQual, split value: 6
100 loops, best of 3: 8.63 ms per loop
```

## 多次决策

```
def find_split(self):
  for c in range(self.c):
    self.find_better_split(c)
  if not self.is_leaf:
#     pdb.set_trace()
    x = self.x.iloc[:, self.var_idx].values
    l_idxs = np.nonzero(x > self.split_val)[0]
    r_idxs = np.nonzero(x <= self.split_val)[0]
    self.lb = DecisionTree(self.x.iloc[l_idxs], self.y[l_idxs], min_leaf=5)
    self.rb = DecisionTree(self.x.iloc[r_idxs], self.y[r_idxs], min_leaf=5)
```

读到这里，如果你有编程经验，那会很容易从决策树联想到二叉树，经典的二叉树是通过递归实现的。这里也是借鉴了递归的思想，在决策树中生成决策树，直到到达叶子节点为止。

```
t = RandomForest(trn_x, trn_y, 1).trees[0]
t.rb, t.lb, t.lb.lb, t.lb.rb

(samples: 617, value: 11.809290068879918, score: 151.00603314209667,
split variable: GrLivArea, split value: 1376,
 samples: 383, value: 12.386518196334503, score: 90.79697982952005,
split variable: OverallQual, split value: 7,
 samples: 162, value: 12.584355513747466, score: 36.72007562937957,
split variable: GrLivArea, split value: 1915,
 samples: 221, value: 12.241497176330432, score: 40.144323458391156,
split variable: GrLivArea, split value: 1935)
```

![](https://upload-images.jianshu.io/upload_images/13575947-9541558097536f99.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从两个模型的测试结果来看，两个模型的决策树分类结果都是相同的，只是在分类值上略有不同。要验证模型的实际预测效果还是需要通过$R^2$ score。

## R^2 Score

```
def predict(self, x):
  return [self.predict_row(xi[1]) for xi in x.iterrows()]

def predict_row(self, xi):
  if self.is_leaf: return self.val
  t = self.rb if xi[self.var_idx] <= self.split_val else self.lb
  return t.predict_row(xi)

DecisionTree.predict = predict
DecisionTree.predict_row = predict_row
metrics.r2_score(t.predict(val_x[:5]), val_y[:5]), metrics.r2_score(m.predict(val_x[:5]), val_y[:5])

(0.9243634901216589, 0.8916108333330888)
```

$R^2$是衡量模型解释数据变化的能力，也就是x变了y会怎么变，它的取值范围是[$-\infty, 1$]，值越大表示模型预测能力越强。我们的模型比对标模型表现得还要好！

## Final model

万事俱备，现在是时候用完整的数据来检验完整的模型，在此之前，需要为随机森林添加predict()。

```
def predict(self, x):
  return np.mean([t.predict(x) for t in self.trees], 0)

RandomForest.predict = predict
```

另外，还记得前文提到的两个随机性吗？采样随机性已经实现了，就剩决策随机性了。所谓的放弃“好决策”拥抱“坏决策”，实际上就是随机地丢弃部分待决策的特征。max_features=0.5的意思就是随机丢弃50%待决策特征。

```
class DecisionTree():
......
  def find_split(self):
#     pdb.set_trace()
    idxs = np.random.permutation(self.c)[:int(self.c * self.max_features)]
    for c in idxs:
      self.find_better_split(c)
......
```

最后，我们用同一个验证集（val_x）分别验证两个模型。可以看到，我们这个不到80行的模型表现得比sklearn的模型还要好！但前者的运行时间却是后者的1400多倍！

```
%time rf = RandomForest(trn_x, trn_y, 20, min_leaf=2, max_features=0.5, sample_sz=800)
metrics.r2_score(rf.predict(val_x), val_y)

CPU times: user 3min 21s, sys: 388 ms, total: 3min 22s
Wall time: 3min 22s
0.8489631454261701
```

```
m = RandomForestRegressor(n_estimators=20, n_jobs=1, min_samples_leaf=2, max_features=0.5)
%time m.fit(trn_x, trn_y)
metrics.r2_score(m.predict(val_x), val_y)

CPU times: user 138 ms, sys: 1.99 ms, total: 140 ms
Wall time: 140 ms
0.8454547075884457
```

## Speed Up & Cython

我们自己编写的模型之所以要比sklearn模型要慢得多，主要的原因在于sklearn是用Cython写的。Python和Cython虽然语法上相差不大，但它们生成的程序有本质区别。Python程序的运行依托于Python解释器，而Cython像C语言一样会编译成机器码，因此Cython程序的运行速度远超Python程序。

通过运行一段简单代码，你就可以看到它们之间在执行速度上质的区别：

```
def foo1(n):
  i = 0
  p = 2
  while i < n:
    i += 1
    p = p * i
%timeit foo1(100)

100000 loops, best of 3: 12.4 µs per loop
```

```
%load_ext Cython
%%cython
def foo2(int n):
  cdef int i = 0
  cdef int p = 2
  while i < n:
    i += 1
    p = p * i
%timeit foo2(100)

10000000 loops, best of 3: 57.9 ns per loop
```

可以看到，对于这个简单到爆的函数，python运行一次需要12皮秒，而编译成Cython程序运行一次只需不到60纳秒，两者相差200倍！

因此，如果你有兴趣，欢迎你用Cython重写这段不到80行的程序，我相信它不会比sklearn模型慢太多。

## END

本文通过从零开始编写随机森林的方式，解析随机森林的核心原理：决策树集成和采样/决策随机化，并深入分析了决策的工作原理和几个核心参数：min_leaf、max_features，最后还简单介绍了通过Cython可以从根本上改善整个模型的运行速度。

