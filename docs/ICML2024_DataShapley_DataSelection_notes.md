# ICML 2024《Rethinking Data Shapley for Data Selection Tasks: Misleads and Merits》公式总结

> 来源文档:仓库根目录 `ICML 2024 Rethinking Data Shapley for Data Selection Tasks Misleads and Merits.doc`(中文翻译版)。
> 注意:该 .doc 的数学公式在转换中丢失(只剩残缺 LaTeX 文本),本文公式按原论文 (Kwon & Zou, ICML 2024) 标准定义重建。
> 文档实际包含两篇论文:主体为 ICML 2024 这篇;文末附带原始 Data Shapley 论文 (Ghorbani & Zou 2019) 译文。
> 整理日期:2026-08-07。

---

## 一、基础设定

**效用函数 (Utility Function)** —— 一切估值的基础

$$v: 2^{N} \to \mathbb{R}, \qquad v(S) := \mathrm{ValAcc}(A(S))$$

- $N = \{1,\dots,n\}$ 是训练集,$2^N$ 是其幂集(所有子集的集合)。
- $A$ 是学习算法(输入数据子集、输出模型),$\mathrm{ValAcc}$ 是模型在验证集上的性能度量(如准确率)。
- 全文统一假设 $v(S) \in [0,1]$、$v(\emptyset) = 0$。
- **用法**:把"一个数据子集好不好"抽象成可计算的数,是所有后续公式的输入。实际使用时 $v$ 通常不可解析求值(必须真训练模型),这是后续问题的根源。

**数据 Shapley 值 (定义 1,Shapley 1953)**

$$\phi_i(v) = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!(n-|S|-1)!}{n!}\,\big[v(S\cup\{i\}) - v(S)\big]$$

- 等价写法:$\phi_i(v) = \frac{1}{n}\sum_{S \subseteq N\setminus\{i\}} \binom{n-1}{|S|}^{-1}\big[v(S\cup\{i\}) - v(S)\big]$。
- **含义**:数据点 $i$ 对所有可能子集的边际贡献 $v(S\cup\{i\}) - v(S)$ 的加权平均。
- **用法**:给每个数据点打分 $\phi(v) = (\phi_1,\dots,\phi_n)$,是唯一同时满足四个公理(对称性、零贡献、可加性等)的数据估值。实际计算用蒙特卡洛置换抽样近似(文中预算 40,000 个效用样本)。

## 二、数据选择任务

**最优数据选择 (式 1)**

$$S^{*}_{(k,v)} = \mathop{\arg\max}_{S \subseteq N,\ |S| = k} v(S)$$

- **用法**:形式化定义"从 $N$ 中选 $k$ 个点使模型性能最优"。直接求解需要遍历 $\binom{n}{k}$ 个子集并逐个训练,不可行。

**Shapley 式选择(代理指标)**

$$\phi(v)[S] := \sum_{i \in S} \phi_i(v), \qquad S^{\phi}_{k}(v) = \text{取 } \phi \text{ 值最大的前 } k \text{ 个点}$$

- **用法**:用"子集内 Shapley 值之和"代替 $v(S)$ 作为子集好坏的代理,实践上就是简单取 Top-$k$。这是 Data Shapley 在数据选择中的标准用法,也是全文质疑的对象。

## 三、理论否定:一般情况下 Shapley 不可靠 (第 3 节)

**假设检验框架** —— 把"比较两个子集的效用"建模为检验问题:

$$H_0:\ v(S_1) \ge v(S_2) \quad \text{vs} \quad H_1:\ v(S_1) < v(S_2)$$

基于 Shapley 的检验是任意函数 $h: \mathbb{R}^n \to [0,1]$,$h(\phi)$ 表示拒绝 $H_0$ 的概率。典型例子(Shapley 式选择隐式采用的):

$$h(\phi) = \mathbb{1}\big[\phi(S_1) < \phi(S_2)\big]$$

满足 $H_0$ / $H_1$ 的效用函数集合:

$$F^{(0)}_{S_1,S_2} := \{v \mid v(S_1) \ge v(S_2)\}, \qquad F^{(a)}_{S_1,S_2} := \{v \mid v(S_1) < v(S_2)\}$$

评估指标(真阳率、真阴率):

$$\mathrm{TruePos}(h) := \mathbb{P}_{v \sim \mathrm{Unif}(F^{(a)})}\big[h(\phi(v)) > \tfrac12\big], \qquad \mathrm{TrueNeg}(h) := \mathbb{P}_{v \sim \mathrm{Unif}(F^{(0)})}\big[h(\phi(v)) < \tfrac12\big]$$

随机猜测 $h \equiv 0.5$ 恰好实现 $\mathrm{TruePos} + \mathrm{TrueNeg} = 1$。

**定理 2(核心负面结论)**:对任意基于 Shapley 的检验 $h$:

$$\mathrm{TruePos}(h) + \mathrm{TrueNeg}(h) \le 1 \quad\Longrightarrow\quad \sup_h \big[\mathrm{TruePos}(h) + \mathrm{TrueNeg}(h)\big] = 1$$

即:不假设效用函数结构时,**Data Shapley 至多与随机选择持平**。

**定理 3(原因)**:对任意评分向量 $s \in \mathbb{R}^n$、任意数据集对 $(S_1,S_2)$,都存在 $v \in F^{(0)}$ 和 $v' \in F^{(a)}$,使得 $s = \phi(v) = \phi(v')$。

- **用法**:Shapley 映射 $v \mapsto \phi(v)$ **不是单射** —— 不同效用可产生完全相同的 Shapley 向量,却对 $S_1$ vs $S_2$ 给出相反排序,因此仅凭 Shapley 值无法可靠比较。证明用到 Shapley 变换零空间的高维性质。
- 注 4:该结论可推广到所有满足"逆帕斯卡三角条件"的半值,包括留一法误差 (Koh & Liang 2017) 和数据 Banzhaf (Wang & Jia 2023)。
- 注 3 的联邦学习例子:$v(S) = |S|/3$ 与另一个 $v'$ 产生相同 Shapley 向量,但对 $\{1,2\}$ vs $\{1,3\}$ 的效用排序相反——差异在 Shapley 层面无法区分。

## 四、什么时候 Shapley 是对的 (第 4 节)

**定理 4(异质质量数据集)**:设 $N = S_{\mathrm{bad}} \cup S_{\mathrm{clean}}$(坏数据 = 错标/噪声,干净数据为其余部分),若

$$\forall\, j \in S_{\mathrm{clean}},\ \forall\, i \in S_{\mathrm{bad}},\ \forall\, S \subseteq N\setminus\{i,j\}:\quad v(S \cup i) \le v(S \cup j)$$

则当 $k = |S_{\mathrm{clean}}|$ 时,Data Shapley 对 $k$ 规模选择**最优**。

- **用法**:解释错标/噪声检测为何是 Shapley 的强项 —— "干净点替换脏点不降性能"这一自然条件即可保证。但该条件只保证特定 $k$ 下的最优性。

**定义 5(单调变换模函数,MTM)** —— 全文最重要的结构假设:

$$v(S) = f\Big(w_0 + \sum_{i \in S} w_i\Big), \qquad f \text{ 单调递增},\ w_i \in \mathbb{R}$$

- 特例:模函数 (Modular) $v(S) = w_0 + \sum_{i\in S} w_i$($f$ 为恒等映射,边际收益为常数、曲率为 0)。
- **定理 6**:对任意 MTM 效用,Data Shapley 对**所有** $k = 1,\dots,n-1$ 都是最优选择。
- **核方法实例**:二分类核方法预测 $\hat{y} = \mathrm{sign}\big(\sum_{i\in S} y_i k(x_i, x_{\mathrm{val}})\big)$,以验证点预测正确性为效用:

$$v(S) = \mathbb{1}\Big[\sum_{i\in S} \underbrace{y_i\, y_{\mathrm{val}}\, k(x_i, x_{\mathrm{val}})}_{=:\, w_i} \ge 0\Big], \qquad f(t) = \mathbb{1}[t \ge 0]$$

即 $v$ 是 MTM。**用法**:MTM 类函数构成"Shapley-有效子空间"的充分条件,涵盖核方法、阈值最近邻等实际学习算法。

## 五、启发式:预测 Shapley 何时有效 (第 5 节)

**MTM 拟合**:用 MTM 函数 $\hat{v}$ 逼近原始效用 $v$。训练数据为子集-效用配对:

$$S_{\mathrm{train}} = \{(S_1, v(S_1)), \dots, (S_m, v(S_m))\}$$

目标:最小化这些配对上的预测误差。**拟合残差**:

$$R_v(\hat{v}) := \mathbb{E}_{S \sim \mathrm{Unif}(N)}\big[(v(S) - \hat{v}(S))^2\big]$$

**归一化拟合残差**(去除效用函数尺度差异):

$$\bar{R}_v(\hat{v}) = \frac{R_v(\hat{v})}{\mathrm{Var}_{S \sim \mathrm{Unif}(N)}[v(S)]}$$

- **用法**:$\bar{R}$ 小 → $v$ 可被 MTM 紧密近似 → 预测 Data Shapley 有效;$\bar{R}$ 大 → 选择性能波动大、不可预测。
- 注 6(零额外成本):拟合 $\hat{v}$ 直接复用估算 Shapley 时已采集的效用样本 $\{(S, v(S))\}$。
- 注 7:MTM 是**充分非必要**条件,残差大不代表 Shapley 一定失效;但中等残差范围内 $\bar{R}$ 与数据选择性能强相关。

**定义 7(ρ-相关数据集对)**:$S \sim \mathrm{Unif}(N)$ 均匀采样,$S'$ 的每个坐标独立地以概率 $(1-\rho)/2$ 与 $S$ 不同(即以概率 $\rho$ 相同)。

**ρ-一致性指数**:

$$\mathrm{Corr}_{\rho}(v) := \mathrm{Corr}\big[v(S),\ v(S')\big], \qquad (S, S') \sim \rho\text{-corr}(N)$$

**定理 8(近似质量 ↔ 一致性)**:设 $M$ 为所有 MTM 函数的集合,对任意效用函数 $v$:

$$\min_{\hat{v} \in M} \bar{R}_v(\hat{v}) \;\le\; \frac{1}{2}\Big(1 - \mathrm{Corr}_{\rho}(v)\Big)$$

- **用法**:效用随数据微小扰动越稳定(一致性指数高),MTM 近似越好,Shapley 越可能有效。这是调和分析中伪布尔函数最优线性逼近上界(基于噪声稳定性,O'Donnell 2014)的推广;$f$ 取恒等时退化为 Saunshi et al. 2022 的定理 3.1。

## 六、实验度量 (第 6 节)

**归一化效用差异**(评价数据选择性能):

$$\mathrm{NUD} = \frac{v\big(S^{\phi}_{k}(v)\big) - v\big(S^{*}_{k,v}\big)}{v\big(S^{*}_{k,v}\big) - \mathbb{E}_{S:|S|=k}\big[v(S)\big]}$$

- **用法**:分子 = Shapley 选择与最优选择的效用差,分母 = 最优与随机平均的效用差。归一化后 1 表示与最优持平,0 表示与随机持平。实践中 $v(S^{*}_{k,v})$ 用随机抽样子集的最大效用近似,期望项用 50,000 个子集的平均效用近似。

**MTM 的神经网络实现**:

$$\hat{v}(S) = f\Big(w_0 + \sum_{i=1}^{n} w_i x_i\Big), \qquad x \in \{0,1\}^n \text{ 为子集指示向量}$$

- 线性组合 $w_0 + \sum w_i x_i$ 由线性层实现;单调函数 $f$ 用**非负权重约束的神经网络**实现。实测大多数数据集拟合 MSE < $10^{-4}$。

**实验要点**:每个数据集生成 200 个噪声变体(噪声率 0–50% 均匀采样),每个变体上评估 Shapley 的 $k$-size 选择性能($k/n \in \{10\%, 30\%, 50\%, 70\%\}$),并训练 MTM 评估 $\bar{R}_v$。结论:小 $\bar{R}$ 恒对应更好的选择;大 $\bar{R}$ 时性能波动显著。

## 七、附:文末附带论文《Data Shapley: Equitable Valuation of Data for Machine Learning》(Ghorbani & Zou 2019)

**数据 Shapley 值 (命题 1)**:由三条公理唯一确定:

$$\phi_i(D,A,V) = \sum_{S \subseteq D\setminus\{i\}} \frac{|S|!(n-|S|-1)!}{n!}\,\big[V(S\cup\{i\}) - V(S)\big]$$

- 对照基准:**留一法** $\phi_i = V(D) - V(D\setminus\{i\})$(不满足公平公理)。
- **三条公理**:
  1. 零贡献 (dummy):若对所有 $S \subseteq D\setminus\{i\}$ 有 $V(S) = V(S\cup\{i\})$,则 $\phi_i = 0$;
  2. 对称性:若 $V(S\cup\{i\}) = V(S\cup\{j\})$ 对所有 $S$ 成立,则 $\phi_i = \phi_j$;
  3. 可加性:$\phi_i(V + W) = \phi_i(V) + \phi_i(W)$。
- 性能评分常写为负损失和:$V = -\sum_{k \in \mathrm{test}} \ell_k$(0-1 损失或交叉熵)。
- **域适应应用**:加权损失重训 $L' = \sum_i \phi_i \ell_i$;流程为计算源数据 Shapley → 删除负值点 → 按相对正值重新加权 → 重训。案例:皮肤病变 29.6% → 37.8%,性别识别 84.1% → 91.5%,临床记录 87.5% → 90.1%。

## 一句话脉络

> **一般情况**:$v \mapsto \phi(v)$ 非单射 → 定理 2/3,Shapley 至多等同随机选择;
> **特殊情况**:效用是 MTM($v(S) = f(w_0 + \sum_{i\in S} w_i)$,如核方法、干净/脏混合数据)→ 定理 4/6,Shapley 对任意 $k$ 最优;
> **实用判断**:拟合残差 $\bar{R}_v$ + ρ-一致性指数 $\mathrm{Corr}_\rho(v)$ → 定理 8,残差小则 Shapley 值得用。

---

## 术语对照速查

| 中文 | 英文 | 记号 |
|------|------|------|
| 效用函数 | utility function | $v: 2^N \to \mathbb{R}$ |
| 数据 Shapley 值 | data Shapley value | $\phi_i(v)$ |
| 最优 $k$ 选择 | optimal $k$-selection | $S^{*}_{(k,v)}$ |
| Shapley 式选择 | Shapley-based selection | $S^{\phi}_{k}(v)$ |
| 单调变换模函数 | monotone transformed modular (MTM) | $v(S) = f(w_0 + \sum_{i\in S} w_i)$ |
| 拟合残差 / 归一化残差 | fit residual / normalized | $R_v(\hat{v}),\ \bar{R}_v(\hat{v})$ |
| ρ-一致性指数 | $\rho$-consistency index | $\mathrm{Corr}_{\rho}(v)$ |
| 归一化效用差异 | normalized utility gap | NUD |
