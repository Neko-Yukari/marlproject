# ExplabOff: Towards Explorative and Collaborative Task Offloading via Mutual Information-Enhanced MARL

## Document Metadata

- **Filename**: ExplabOff_Towards_Explorative_and_Collaborative_Task_Offloading_via_Mutual_Information-Enhanced_MARL.pdf
- **Pages**: 10
- **Subject**: IEEE INFOCOM 2025 - IEEE Conference on Computer Communications;2025; ; ;10.1109/INFOCOM55648.2025.11044758
- **Total Characters**: 59868
- **Avg Chars/Page**: 5986.8

---

## Page 1

ExplabOff: Towards Explorative and Collaborative Task
Offloading via Mutual Information-Enhanced MARL
Tao Ren*, Zheyuan Hu†, Jianwei Niu†‡§, Yiming Yao†
* State Key Laboratory of Intelligent Game, Institute of Software Chinese Academy of Sciences,
University of Chinese Academy of Sciences, Beijing, China
† State Key Laboratory of Virtual Reality Technology and Systems, School of Computer Science and Engineering,
Beihang University, Beijing, China;
‡ Zhongguancun Laboratory, Beijing, China
§ Zhengzhou University Research Institute of Industrial Technology, Zhengzhou University, Zhengzhou, China
Email: rentao22@iscas.ac.cn, {huzheyuan18, niujianwei, yaoyiming}@buaa.edu.cn
Abstract—Multi-access edge computing provides mobile de-
vices (MDs) with both satisfactory computing resources and
task latency, by ofﬂoading MDs’ tasks to nearby edge servers.
There is a popular trend to develop decentralized ofﬂoad-
ing (dec-ofﬂoading) approaches using multi-agent reinforcement
learning (MARL), primarily based on centralized-training and
decentralized-execution. However, the dec-ofﬂoading policies to-
gether could also lack exploration and collaboration since each
MD is guided by the policy-critic only through ofﬂoading costs
without explicitly considering the impacts of other MDs’ ofﬂoad-
ing behaviors. Motivated by this, we propose Explorative and
collaborative Offloading (ExplabOff) that can achieve superior
dec-ofﬂoading by consciously exploiting the implicit exploration
and collaboration information involved in MDs’ states and
actions. Speciﬁcally, we design two additional policy-learning
metrics, the exploration-metric based on the maximum entropy
of MDs’ joint ofﬂoading actions and collaboration-metric based
on one MD’s belief about others’ ofﬂoading behaviors. Then, we
assemble these metrics into a new criterion deﬁned as the mutual
information (MI) between MDs’ states and actions, and adopt
it as an additive reward except for the vanilla reward during
centralized-training. Furthermore, we distinguish MI between
superior and inferior ofﬂoading, strengthening and weakening
them discriminatively. Experiments on both simulation and real-
testbed verify the effectiveness of ExplabOff over state-of-the-art
dec-ofﬂoading.
Index Terms—Task Ofﬂoading, Multi-Access Edge Computing,
Decentralized Ofﬂoading, Multi-Agent Reinforcement Learning
I. INTRODUCTION
Recent years have witnessed a rapid development of
mobile devices (MDs), along with the explosive growth of
mobile applications [1], e.g., augmented reality, online gaming
and autonomous driving. These applications are typically
computation-intensive and latency-critical, resulting in the
emergence of multi-access edge computing (MEC) that can
provide MDs with both satisfactory computing resources and
task latency by deploying edge servers (ESs) in base stations
(BSs) in close proximity to MDs [2]. One of the key issues in
MEC is task ofﬂoading that decides whether, where and how
much to ofﬂoad MDs’ tasks in each system slot, so that the
whole performance of MEC could be maximized [3], [4].
This work was supported in part by National Key R&D Program of China under
Grant No. 2023YFB4503700, National Natural Science Foundation of China
under Grant No. U23B2025, 62372027. (Corresponding author: Jianwei Niu.)
Great efforts have been devoted to developing efficient
offloading algorithms, lots of which are based on mathematical
programming (MP) and achieve offloading solutions in closed
form [5], [6]. Whereas, owing to the ever-increasing transmitting
frequency, it becomes more and more difficult for MP-based
offloading to obtain solutions in limited time, especially when
offloading variables are of high dimension [7]. Therefore, there
is a popular trend to design offloading algorithms based on (deep)
reinforcement learning (RL), which transforms the multi-slot
offloading problem into Markov decision process (MDP) and
solves it using RL by setting cost-minimization as rewards [8].
A large variety of works have focused on developing RL-
based offloading policies [9], [10], but most consider centralized
offloading where the offloading policy is deployed in a controller
that collects global states (task sizes, left energy, etc.) from and
dispatches offloading decisions (which ES, task proportion, etc.)
to all MDs [11], [12]. In practice, when MDs are of large size,
centralized offloading could also face troubles due to the heavy
burden of collecting all MD’s states for each decision-making
[13], thus stimulating the increasing interest on decentralized
offloading (dec-offloading) [4], [14] primarily based on multi-
agent RL (MARL), such as [15]–[17] based on MADDPG, [18]
based on COMA, [19] based on MAPPO, and [20] based on
MATD3. In dec-offloading where each MD makes decisions
based on local states as shown in Fig. 1a, studies focus on not only
efficient explorations of joint offloading spaces but also superior
collaborations of MDs’ offloading behaviors. Some works [21],
[22] assume inter-MD communication to facilitate exploration and
collaboration which could be inapplicable in realistic scenarios
where D2D communication is not permitted or available, thereby
most studies [18], [23] adopt the paradigm of centralized training
and decentralized execution (CTDE) to learn MDs’ offloading
policies, each of which is guided by a critic granted access to
global states during training but makes offloading decisions only
based on its local states during execution.
However, given all states to the critic, the MDs’ offloading
policies as a whole could also lack exploration and collabo-
ration, since each MD is guided by the critic only through
system costs without explicitly considering the impacts of other
MDs’ offloading behaviors on the explorative and collaborative
performance of the joint policy. Taking the simple MEC in
IEEE INFOCOM 2025 - IEEE Conference on Computer Communications | 979-8-3315-4305-1/25/$31.00 ©2025 IEEE | DOI: 10.1109/INFOCOM55648.2025.11044758
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 2

CPU: 6 ×109
optimal collaboration
sub-optimal collaboration 2
sub-optimal collaboration 1
task: 5+1 ×109
MD2
MD1
MD3
task: 2+1 ×109
task: 6+1 ×109
ES1
ES2
CPU: 12 ×109
MD1/2/3 CPU: 1 ×109
(a) Decentralized-ofﬂoading MEC.
0
1
2
3
4
5
6
7
8
9
1
2
E S
S l o t
 M
D  3
0
1
2
3
4
5
6
7
8
9
1
2
E S
 M
D  2
0
1
2
3
4
5
6
7
8
9
1
2
E S
 M
D  1
(b) MD 1/2/3 ofﬂoads tasks to ES 1/2.
MD1
MD2
MD3
0    1    2    3    4    5    6    7    8    9   10
Slot  (offloading to ES1)
10
9
8
7
6
5
4
3
2
1
0
Slot  (offloading to ES2)
(c) MDs’ ofﬂoading explorations.
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
2
4
6
M
D 3 - > E S 1 ,  M
D 1 / 2 - > E S 2
C o s t
E p i s o d e
 M
A R L - t r a i n e d  o f f l o a d i n g
 O p t i m a l  o f f l o a d i n g
M
D 2 - > E S 1 ,  M
D 1 / 3 - > E S 2
(d) MARL-trained ofﬂoading cost.
Fig. 1: A pilot experiment on existing MARL-based dec-ofﬂoading.
Fig. 1a as a motivation example, where computing capacities
of MDs/ES1/ES2 are 1/6/12 G cycles per slot, computing
requirements of tasks arriving at MD1/2/3 each slot follow the
normal distribution around 7/6/3 G cycles, and ES1/ES2 are
equipped with task buffers. Adopting MADDPG as in [15]–[17]
to learn dec-offloading policies for MDs, it is found: (1) One
MD’s offloading decisions, e.g., MD2 chooses ES1 in most slots1
in Fig. 1b, happen to coincide with MD3, which could cause
overloads of ES1 because of lacking collaboration2; (2) Joint
offloading behaviors of all MDs tend to explore within limited
spaces during training, as shown in Fig. 1c where offloading
decisions (to ES2) of MD1/2 (blue and red points) lack high
coverage, which could hinder learning efficiency and superiority;
(3) System costs of MARL-trained offloading are inclined to
reach the sub-optimal, i.e., the red line corresponding to the
joint behavior {MD3→ES1, MDs1/2→ES2} in Fig. 1d, while
the optimal is obviously the blue line of the joint behavior
{MD2→ES1, MDs1/3→ES2}. Possible reasons could lie in two
aspects: One is the limited capability of exploring more superior
joint offloading behaviors during centralized training signaled
only by system costs; The other is the lack of each MD’s strong
perception on other MDs’ offloading behaviors to pursue more
desirable system-level collaboration.
Facing the issues, we propose a more Explorative and
collaborative task Offloading (ExplabOff) approach, that can
achieve superior joint offloading by consciously exploiting the
implicit exploration and collaboration information involved in
MDs’ states and behaviors. First, we formulate the optimization
problem for task offloading with the objective of minimizing
system costs, and transform it into a MDP problem suited to
solve using MARL for dec-offloading. Second, we design two
additional optimizing metrics, i.e., the exploration metric based
on the maximum entropy of MDs’ joint offloading policy, and the
collaboration metric based on one MD’s belief about other MDs’
1Due to the existence of task buffers, offloading tasks to the same ES is possible.
2It’s more promising to see MD2 choose ES1 and MD1/3 choose ES2.
offloading behaviors. Then, we assemble the two metrics into a
new criterion defined as the mutual information (MI) between
MDs’ states and behaviors, which is adopted as an additive
objective except for original rewards during centralized training.
Next, in view of the possible suboptimality caused by blindly
optimizing MI, we propose to distinguish MI between superior
and inferior offloading episodes and try to promote MI in supe-
rior episodes and weaken in inferior ones. Moreover, concerning
the intractability in directly computing MI, we adopt the neural
estimators InfoNCE [24] and L1Out [25] to estimate the lower
and upper bounds of MI in superior and inferior episodes for
maximization and minimization, respectively. Finally, extensive
experiments on both simulation and testbed are conducted to
verify the advantages of ExplabOff over state-of-the-art MARL-
based offloading.
Our main contributions are summarized as follows:
• We design the exploration and collaboration metrics as
additional ofﬂoading objectives except for system costs, to
help MDs learn joint ofﬂoading policies more exploratively
and collaboratively during centralized training.
• We combine the exploration and collaboration metrics into
a MI-enhanced objective to guide the learning of offloading
policies, and further distinguish MI between superior and inferior
offloading experiences in case of blindly enhancing MI.
• We propose to strengthen and weaken MI for superior and
inferior ofﬂoading experiences, respectively, and further
adopt the neural estimators InfoNCE and L1Out to maximize
and minimize superior and inferior MIs.
• We conduct extensive experiments to verify the advantages of
ExplabOff over state-of-the-art MARL-based offloading via
both simulation and testbed, and also provide revealing insights
into the efficacy of ExplabOff via various visualization analysis.
II. SYSTEM MODEL
We consider a MEC system consisting of E ESs (E =
{1, 2, ..., E}) and M MDs (M = {1, 2, ..., M}). The system
time is equally divided into N ={1, 2, ..., N} slots. For each
ES e∈E, the computing capacity is denoted as f e, and a task
queue qe is deployed with qe
n being the queue size at slot n.
For each MD m ∈M, the computing capacity is denoted as
f m, and the energy budget is denoted as εm with εm
n being
the available energy at slot n. Besides, for each MD m, a task
jm
n ={dm
n , c, tmax} (where dm
n , c and tmax represent the task size,
required amount of CPU cycles per bit, and maximum tolerable
latency, respectively) arrives at each slot n, and a task offloader
πm is maintained to partially3 offload jm
n to an appropriate ES at
each slot n. The ofﬂoader πm decides the ofﬂoading proportion
ρm
n and destination ϵm
n for jm
n using a variable,
am
n = {ρm
n , ϵm
n },
s.t.
ρm
n ∈[0, 1],
(1)
ϵm
n ∈E,
if ρm
n ̸= 0.
(2)
3Similar to [26], partial offloading is considered, which could also easily extend
to binary offloading by constraining the offloading proportion to 0 or 1.
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 3

Due to unavailable D2D communications in most scenarios
[3], ofﬂoaders work independently but also expect to jointly
minimize system costs. A MEC with two ESs is given in Fig. 2.
ES2
MD2
MD m
MD1
ES’s computing capacity 𝑓𝑒
ES’s task queue size 𝑞𝑛e at slot n
Computing capacity 𝑓𝑚 of MD m
Left energy budget 𝜀𝑛𝑚of MD m at slot n
MD’s task 𝑗𝑛𝑚at slot n
MD’s offloader πm
Partial offloading of MD’s task to ES
ES1
MD M
Fig. 2: A MEC with two ESs and M MDs. At each slot n, each MD’s
offloader πm independently chooses to partially offload its task jm
n to
one ES and computes the left locally. All offloaders expect to collaborate
on task offloading, so as to minimize long-term system costs.
A. Communication Model
Similar to [27], the block fading model is taken to represent
the channel gain gm,e
n
= |ˇfm,e
n
|2 ˆfm,e
n
of the ofﬂoading link
Lm→e
n
from MD m to ES e at slot n, where ˇf and ˆf denote
the small-scale fading4 and large-scale fading5, respectively.
On account of the existence of multiple MD-ES ofﬂoading
links at each slot n, the channel interference Im,e
n
of the link
Lm→e
n
suffered from other links is represented as
Im,e
n
=P
m′∈M,m′̸=m1(ρm′
n ̸=0) ptran gm′,e
n
,
(3)
where ptran is MDs’ transmitting power. Hence, the transmission
rate υm,e
n
of the ofﬂoading link Lm→e
n
at slot n is modeled as
υm,e
n
=B log2(1+ptrangm,e
n
/(Im,e
n
+σ2)),
(4)
where B is the system bandwidth and σ2 is the Gaussian noise.
B. Edge Computing Model
If ρm
n
̸= 0, the ρm
n part of jm
n
is ofﬂoaded to the ES
e = ϵm
n for edge computing, whose latency includes: 1) The
transmitting time tm,tran
n
of ρm
n from MD m to ES e, which
could be calculated as tm,tran
n
= ρm
n dm
n /υm,e
n
. 2) The waiting
time tm,wait
n
of ρm
n for both tasks arriving on qe before ρm
n at
slot n and tasks left in qe at the beginning of slot n, that can
be calculated as tm,wait
n
=
  P
m′∈M′ ρm′
n dm′
n +qe
n

c/f e where
M′={m′| m′∈M, m′̸=m, ρm′
n ̸=0, ϵm′
n =e, tm′,tran
n
<tm,tran
n
} is the
MDs ﬁnishing transmission to the same ES e before MD m.
3) The startup time for the ﬁrst execution6 of MD m’s tasks on
ES e during an episode [31], [32], which is assumed to be tm,e
start .
4) The executing time tm,exe
n
of ρm
n on ES e, which can be
calculated as tm,exe
n
= ρm
n dm
n c/f e.
With the above, the task latency tm,edge
n
and energy consump-
tion em,edge
n
of MD m for edge-computing ρm
n can be given by
tm,edge
n
=tm,tran
n
+tm,wait
n
+1(∀n′<n, ϵm
n′̸=e) tm,e
start +tm,exe
n
,
(5)
4According to the Jakes’ model [28], the small-scale fading ˇf is modeled
as a ﬁrst-order Gaussian Markov process whose detail could be found in [29].
5Based on the LTE standard, the large-scale fading ˆf, involving path-loss and log-
normal shadowing as in [30], is modeled as ˆfm,e
n
=−148.1−37.6 lg dm,e
n +10 lg z,
where dm,e
n
is the distance of Lm→e
n
and z is the log-normal variable.
6Each ES e only needs to conduct startup preparation for MD m’s tasks
once (the ﬁrst task jm
n of MD m ofﬂoaded to e, i.e., ∀n′<n, ϵm
n′ ̸= e).
em,edge
n
= ptranρm
n dm
n /υm,e
n
.
(6)
C. Local Computing Model
If ρm
n ̸= 1, the 1 −ρm
n part of jm
n is computed by MD m
locally. The local task latency tm,loc
n
of 1 −ρm
n is calculated as
tm,loc
n
= (1 −ρm
n )dm
n c/f m,
(7)
and the corresponding local energy consumption em,loc
n
is given by
em,loc
n
= ξ(f m)2(1 −ρm
n )dm
n c,
(8)
where ξ is the energy efﬁciency coefﬁcient of MDs’ chips [33].
D. System Cost Model
1) Time Cost: Since jm
n could be executed simultaneously
on MDm (ρm
n ̸= 1) and ES (ρm
n ̸= 0), the task latency tm
n of
jm
n is the maximum time on two venues that can be given by
tm
n = max{tm,loc
n
, tm,edge
n
},
(9)
which should not exceed the maximum tolerable latency, i.e.,
tm
n ≤tmax.
(10)
2) Energy Cost: Different from the time cost, the consumed
energy em
n of jm
n should include the energy consumption of
MD m for both edge computing and local computing, i.e.,
em
n = em,edge
n
+ em,loc
n
.
(11)
Then for ∀n ∈N, the available energy budget εm
n of MD
m can be given by εm
n = εm
n−1 −em
n , which should satisfy
εm
n ≥0.
(12)
3) System Cost: The task cost cm
n for executing jm
n is given
by cm
n =ηtm
n +(1−η)em
n , where η is the weight of time cost.
Hence, the average cost for n is calculated as cn = 1
M
PM
m=1 cm
n ,
and the average system cost c over N slots is given by
c = PN
n=1cn/N.
(13)
III. PROBLEM FORMULATION AND TRANSFORMATION
Viewing the ofﬂoading indicator ρm
n and destination ϵm
n as
optimizing variables, c can be considered as a function about
ρm
n and ϵm
n , i.e., c(ρm
n , ϵm
n ). With the goal of completing the
whole episode with the average system cost as low as possible,
we formulate the optimization problem as
P :
min
ρm
n ,ϵm
n c (ρm
n , ϵm
n ),
(14)
s.t.
Eqs. (1), (2), (10), (12).
Theorem 1. The problem P is NP-hard.
Proof. The main idea of the proof is to draw an analogy
between the problem P and the Generalized Assignment
Problem (Section 15.8.5 in [34]), where M tasks need to be
assigned to E servers, the tasks assigned to any server should
not exceed its capacity, and the cost is required to be minimized.
Detailed proof is omitted due to space limitations.
According to Eq. (13), min c in Eq. (14) aims to minimize
accumulated costs cn over N slots. Thus, by seeming costs
inversely correlated with rewards, the main goal of problem
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 4

P is in line with the objective of discrete-time MDP that
aims to maximize accumulated rewards over multiple time
steps [35]. Therefore, we adopt a MDP, deﬁned by a 5-tuple
(sn, an, P, γ, rn), to reformulate P as follows,
P′ : max
an
1
N
PN
n=1 γn−1r(sn, an),
(15)
where sn = {dm
n , qe
n, εm
n | m ∈M, e ∈E} is the system state7
at slot n, an ={ρm
n , ϵm
n | m∈M} is the actions generated by
all ofﬂoaders πm at slot n, P(sn+1|sn, an) is the probability
(usually unknown as a prior) for MEC transiting from sn
to sn+1 after taking an, r(sn, an) is the reward function that
generates the reward at slot n according to the positive/negative
revenues related to system costs and constraints after taking
an in sn, and γ is the discount factor of future rewards.
In addition, considering the constraints (10) and (12)8 in
Eq. (14), the offloaders πm are expected to complete the whole
episode without violating any constraint, and also achieve the cost
as low as possible. Hence, we define the reward function: (1) When
all πm succeed the whole episode obeying all constraints, they
receive a positive reward inversely correlated with c; (2) When
a πm fails to complete an episode due to violating constraints, it
receives a negative penalty depending on the extent of violation.
IV. OFFLOADING ALGORITHM
Concerning the heavy burden of collecting all MDs’ states
for sn at every slot and the poor scalability of centralized
scheduling, it is more desirable to perform dec-offloading based
on MARL, which primarily adopts the CTDE paradigm to
learn collaborated offloaders [16], [18]–[20]. During centralized
training, the learning of each offloader πm
φ (parameterized by φ)
is guided by a central critic Qθ (parameterized by θ and permitted
access to other MDs’ states as well as actions) with the objective
J(Q) = PN
n=1 Esn∼P, an∼π1,...,πM γn−1r(sn, an).
During the decentralized execution, each πm
φ makes its offloading
decision am
n = {ρm
n , ϵm
n } independently based on its own state
sm
n = {dm
n , qe
n, εm
n }9. However, even all states and actions are
used by Qθ to guide the learning of individual offloader, the
optimization of dec-offloading policies by existing MARL-based
methods only through rewards could still be insufficient: On one
hand, offloaders could be guided by the critic to obtain sub-optimal
offloading policies due to lacking the ability to explore more
potential offloading behaviors. On the other hand, the access of
the critic to all MDs’ states and actions could still not be maximally
exploited to achieve optimal collaboration of joint offloading.
Therefore, we propose ExplabOff10, as shown in Fig. 3, to
promote both exploration and collaboration of MDs’ joint
ofﬂoading behaviors more consciously by exploiting the
implicit information involved in MDs’ states and actions. First,
we design two additional optimizing metrics for problem P′ to
7Time-invariant variables, e.g., fe, fm, c, tmax, ptran, B, are omitted for brevity.
8The constraints (1) and (2) could be directly imposed on the output format of an.
9qe
n is updated for m during each result-downloading of its offloaded task from e.
10Although a simple multi-ES multi-MD MEC is considered in this work for
brevity, more complex MEC scenarios, e.g., wireless power transfer-assisted MEC,
air/space server-enabled MEC, could also be applicable for ExplabOff, where the
differences primarily lie on the problem formulation and transformation of Section III.
enhance the exploration and collaboration of joint ofﬂoading,
respectively. Then, we assemble the optimizing metrics into
a new criterion deﬁned as the mutual information I(sn; an)
between MDs’ states and actions. Next, to avoid optimizing
I(sn; an) blindly, we propose to distinguish I(sn; an) between
superior and inferior ofﬂoading episodes and try to promote
I(sn; an) in superior episodes and weaken I(sn; an) in inferior
ones. Last, concerning the intractability in directly computing
I(sn; an), we adopt the neural estimator InfoNCE and L1Out
to estimate the lower and upper bounds of MI in superior and
inferior episodes for promoting and weakening, respectively.
MEC Environment
Superior Episodes
Experience Buffer
Inferior Episodes
INCE
IL1Out
MD1
𝑠𝑛
1
𝑎𝑛
1
MDM
𝑠𝑛
𝑀
𝑎𝑛
𝑀
rn
Ƹ𝑟𝑛
In
(𝑠𝑛,𝑎𝑛, Ƹ𝑟𝑛, 𝑠𝑛+1)
each slot
each episode
𝑠𝑖, 𝑎𝑖, Ƹ𝑟𝑖, 𝑠𝑖+1
update
𝑠𝑖, 𝑎𝑖
train
……
MD1
𝜋1
𝑄1
𝑄′1
𝜋′1
Centralized Training
train
𝜋𝑀
𝑄𝑀
𝑄′𝑀
𝜋′𝑀
MDM
𝑎′1
𝑎′𝑀
𝑎𝑀
𝑎1
…
…
…
…
Decentralized 
Executing
……
……
update
sample
𝐿𝜋
𝑀
𝐿𝜋
1
𝐿𝑄
1
𝐿𝑄
𝑀
(𝑠1,𝑎1)
(𝑠2,𝑎2)
𝑠𝑖, 𝑎𝑖
sample
each slot
(𝑠𝑛,𝑎𝑛)
Fig. 3: The algorithm overview of ExplabOff.
A. Explorative and Collaborative Ofﬂoading Metrics
In problem P′, the obtained joint ofﬂoading policy π
that generates joint action an
=
{a1
n, a2
n, ..., aM
n |am
n
∼
πm(sm
n ), m ∈M} via CTDE-based MARL could also be
limited in exploration and collaboration by solely depending
on the guidance of the central critic using system rewards
r. Therefore, we design two metrics to further improve the
efﬁciency of the joint ofﬂoading policy π.
Exploration Metric: To encourage the joint ofﬂoading policy
π to explore diverse ofﬂoading actions an in the joint ofﬂoading
space π1 × π2 × ... × πM, we introduce the maximum entropy
term H(an) [36], [37] of the joint ofﬂoading policy π as an
additional guidance of the central critic, which could help
generate diverse joint ofﬂoading actions and avoid the collapse
of joint ofﬂoading behaviors when maximized. Hence, the
objective of Q becomes
J(Q) = P Esn∼P,an∼π γn−1[r(sn, an) + H(an)].
Collaboration Metric: To promote collaboration between
ofﬂoaders πm, m ∈M, it is beneﬁcial to improve the belief of
one MD’s πm about other MDs’ ofﬂoading behaviors during
centralized training, i.e., when πm makes its ofﬂoading decision
am
n given access to all MDs’ states sn it should be as certain as
possible about the decisions a−m
n
made by other MDs, so we
introduce the minimum entropy term −H(a−m
n
|am
n , sn) into
the objective of Q as follows,
J(Q) = P Esn∼P,an∼π γn−1[r(sn, an) −H(a−m
n
|am
n , sn)].
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 5

In addition, considering the belief of each MD itself about its
own decision am
n given all MDs’ states sn during centralized
training, we could also integrate the minimum entropy term
−H(am
n |sn) into the objective of Q, i.e.,
J(Q) = P Esn∼P,an∼π γn−1[r(sn, an) −H(am
n |sn)].
B. Intrinsic Mutual Information-based Ofﬂoading Reward
Taking the above three entropy terms together, we can get11
J(Q)=
X
Eγn−1
r(sn,an)+H(an)−H(a−m
n |am
n,sn)−H(am
n |sn)

.
According to the relationship between conditional and joint
entropy, the last three terms in the above J(Q) can be rewritten as
H(a) −H(a−m|am, s) −H(am|s)
=H(a) −
 H(a−m, am, s) −H(am, s)

−H(am|s)
=H(a) −H(a, s) + H(am, s) −H(am|s)
=H(a) −H(a, s) + H(s)
=I(a; s)
This coincides with the observation that the ofﬂoading
actions generated by well-behaved MDs are mostly coupled
with certain states. Hence, we obtain the ﬁnal form for the
objective of Q as follows,
J(Q) = P Eγn−1
r(sn, an) + I(an; sn)

.
(16)
As shown in Fig. 3, since each MD’s critic Qm is endowed with
the sampled experiences {sn, an, rn, sn+1} containing all MD
states sn and actions an during centralized training, each MD’s
πm could be trained under the guidance of the above Q objective
to be more explorative and collaborative in joint offloading actions
an, as well as more deterministic about its own action am
n .
C. Distinguished Mutual Information Estimation
In MEC environments, owing to the stochasticity of system dy-
namics along with the volatile learning process of MDs’ offload-
ers, the experiences (sn, an, rn, sn+1) consisting of offloading
actions with diverse rn are gathered during MARL. Hence, it
results in the possibility that large I(sn; an) means strong collab-
oration, but could not consequentially lead to large rewards. As
shown by a motivation MEC example in the left of Fig. 4, the red
joint offloading (i.e., MD2→ES1, MDs1, 3→ES2) is the optimal
collaboration while the gray (i.e., MD1→ES1, MDs2, 3→ES2)
and orange (i.e., MD3→ES1, MDs1, 2→ES2) joint offloadings
are sub-optimal collaborations. If all offloading experiences
(red, orange and blue circles, along with gray circles denoting
other joint offloadings) are directly stored in the experience
buffer (right top of Fig. 4) and later sampled for maximizing
I(an; sn), the collaboration leading to both optimal policies and
sub-optimal policies are all strengthened without distinction.
Therefore, according to the MI regularized MARL works
[38]–[40], we further divide ofﬂoading experiences into superior
and inferior ones, stored in the superior buffer B+ and inferior
buffer B−separately as shown in the right bottom of Fig. 4.
By maximizing and minimizing I(sn; an) in B+ and B−
separately, we can persistently strengthen and break the joint
11The subscript {sn∼P, an∼π} of E will be omitted for brevity.
Experience Buffer
𝑓1=7 ×109
optimal collaboration
sub-optimal collaboration 2
The required computing cycles for tasks of MD1/2/3 in each 
slot are assumed to be 7/5/2×109. The computing capacities of 
ES1/2 in each slot are assumed to be 7/14×109.
sub-optimal collaboration 1
𝑑2𝑐=5 ×109
Superior Buffer
Inferior Buffer
𝐼(𝑠;𝑎)
max MI to enhance 
optimal collaborations
min MI to break sub-
optimal collaborations
optimal experiences
other experiences
sub-optimal experience 1
sub-optimal experience 2
MD2
MD1
MD3
𝑑3𝑐=2 ×109
𝑑1𝑐=7 ×109
ES1
ES2
𝑓2=14 ×109
Fig. 4: A motivation MEC example to observe optimal and sub-optimal
joint ofﬂoading collaborations that require distinguishment.
ofﬂoadings corresponding to optimal and sub-optimal collabora-
tions, respectively. Speciﬁcally, we measure the superiority and
inferiority of a piece of experience according to the accumulated
rewards rΓ of the episode Γ it is in. Suppose the lowest episode-
reward in B+ is rmin, then the experiences in Γ are stored into
B+ and the experiences corresponding to the lowest episode-
reward are dequeued if rΓ > rmin. This measure ensures that
B+ always maintains the optimal experiences. If rΓ ≤rmin,
the experiences are stored into B−. The difference is that B−
is kept in a FIFO manner since the most recent sub-optimal
collaborations are eager to break.
D. Mutual Information Lower and Upper Bounds
Concerning the intractability in directly computing MI be-
tween variables (especially for continuous variable with unknown
distribution) [41], we leverage the InfoNCE [24] and L1Out [25]
to estimate the lower bound of MI in B+ for maximization and
upper bound of MI in B−for minimization, respectively.
The lower bound based on state-action pairs (si, ai)∼B+
is approximated using InfoNCE as follows,
I(s; a) ≥INCE(s; a) = log(K) −LNCE,
where LNCE = −E(si,ai)∼B+
h
log
exp(qψN(si, ai))
Esj∼B+exp(qψN(si, aj))
i
.
Here, q(·, ·) is a score function parameterized by ψN, and K
is the number of samples for calculating the expectation E. By
optimizing ψN with the objective of minimizing LNCE, we can
tighten INCE to approach the lower bound of MI.
Similarly, the upper bound based on state-action pairs
(si, ai) ∼B−is approximated using L1Out as follows,
I(s; a)≤IL1Out(s; a)
=E(si,ai)∼B−
h
logqψL1(ai|si)−log Esj̸=si,sj∼B−qψL1(ai|sj)
i
,
where q(·|·), parameterized by ψL1, is a variational approxi-
mation to the real conditional distribution p(ai|si). Similarly,
we can tighten IL1Out to approach the upper bound of MI, by
optimizing ψL1 towards the minimum of IL1Out.
To guide joint ofﬂoading towards superior collaborations
and away from inferior ones, we ﬁrst combine INCE and IL1Out
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 6

into a synthetic12 MI
I(s; a) = µINCE(s; a) −νIL1Out(s; a)
for measuring the superiority/inferiority of an ofﬂoading state-
action pair (s, a) and then integrate I(s; a) as a MI-enhanced
reward into J(Q) in Eq. (16) as
J(Q) = P Eγn−1
r(s, a)+µINCE(s; a)−νIL1Out(s; a)

, (17)
by which we expect to award superior and punish inferior of-
ﬂoading behaviors, respectively, since only superior and inferior
behaviors can obtain high INCE(s; a) and low −IL1Out(s; a).
Algorithm 1 The training algorithm of ExplabOff
1: Initialize parameters θm of critics Q and φm of ofﬂoaders π, and
corresponding target networks θ′m and φ′m, m ∈M
2: Initialize the parameters ψN and ψC of INCE and IL1Out
3: Initialize experience buffer B, superior and inferior buffers B+, B−
4: repeat
5:
Initialize the MEC environment Env
6:
for each slot n in episode Γ do
7:
Each MD observes its state sm
n and generates am
n = πm
φ (sm
n )
8:
Env executes an = {a1
n, ..., a2
n} and generates rn
9:
Store {sn, an, ˆrn, sn+1} into B, where ˆrn =rn+I(sn; an)
10:
end for
11:
Compute the episode-reward rΓ
12:
Store Γ′={sn, an}N
n=1 into
(
B+,
if
rΓ >rmin
B−,
else
13:
Periodically update ψN and ψC via LNCE and LL1Out using
samples from B+ and B−, respectively
14:
For m ∈M, update θm and φm via Lθ(Qm) and ▽φJ(πm)
using samples from B
15:
Periodically update target parameters θ′m and φ′m, m ∈M
16: until convergence
17: return πm, m ∈M
E. Overall Algorithm
To this end, we detail the overall training algorithm of
ExplabOff in Algorithm 1. First (Lines 6-10), in each episode,
each MD m performs dec-ofﬂoading based on its own policy
πm and state sm
n , after which the joint ofﬂoading experience is
stored into B. Then (Lines 11-12), the ofﬂoading state-action
pairs Γ′ of the episode are collected and stored into B+ or B−
according to the episode-reward rΓ. Next (Line 13), the parame-
ters ψN and ψC of INCE and IL1Out are periodically updated with
samples from B+ and B−, respectively, estimating the latest
superior and inferior ofﬂoading collaborations. Consequently
(Line 14), the parameters θm and φm of each MD’s critic Qm
and ofﬂoader πm are updated using samples from B according
to Lθ(Qm) and ▽φJ(πm) as follows,
Lθ(Qm) = Esi,ai,ˆri,si+1∼B

Qm(si, ai) −ˆym
i

,
▽φJ(πm)=Esi,ai∼B

▽φπm(sm
i ) ▽amQm(si,a1
i,...,aM
i )|am
i =πm(sm
i )

,
where ˆym
i = ˆri+γQm
θ′(si+1, a′
i+1)|a′
i+1={π1
φ′(s1
i+1),...,πM
φ′ (sM
i+1)}
and ˆri =ri + I(si; ai). Last (Line 15), the parameters θ′m and
φ′m of target critics and ofﬂoaders are periodically updated.
12µ and ν are hyperparameters that balance the importance of the synthetic
MI-enhanced reward I(s; a) w.r.t. the original reward r(s, a).
V. PERFORMANCE EVALUATION
A. Simulation Setup
1) MEC and ExplabOff Settings: We take a 0.5×0.5 km2
square of the Manhattan city as the investigated area, along
with the MDs analogous to mobile device-holding walkers or
vehicles moving along roads following the Manhattan mobility
model [42], as shown in Fig. 5. ESs are located according to the
data from Opencellid [43]. We investigate three different MEC
settings, including 2ES-3MD (ES1: 6GHz, ES2: 12 GHz), 2ES-
5MD (ES1: 10GHz, ES2: 19 GHz), 3ES-7MD (ES1: 10GHz,
ES2: 19 GHz, ES3: 26 GHz). The task sizes of MDs 1-7 follow
Gaussian distributions, with mean value being 7/6/3/4/5/4.5/5.5
kb and variance 1 kb. The numeral simulation of the MEC
system is built using python, and primary MEC parameters
are listed in Table I unless otherwise speciﬁed. For ExplabOff,
the discount reward factor is 0.99, the MI weights µ and ν are
3.5 and 1, and the superior/inferior buffer size is 1000.
Fig. 5: The investigated area of Manhattan city map.
TABLE I: Primary MEC Parameters
Parameter
Description
Value
fm
MD computing capacity
1 GHz
N
Number of slots
10
δ
Slot length
1 s
c
CPU cycles/bit
900
tmax
maximum task delay
1 s
ptran
MD transmitting power
0.1 W
ξ
Energy efﬁciency coefﬁcient
10−27
σ2
White Gaussian noise
-100 dBm
2) Performance Metrics and Baselines: Performance eval-
uation metrics include two types. (1) The first is the average
task cost (AvgCost) c defined in Eq. (13) of all success episodes
that represents the quality of service of all tasks in MEC. (2)
The second is the task success rate (SuccRate), calculated by
P
n I(tm,Γ
n
<tmax | ∀m∈M)/N, that denotes the extent to which
all tasks are finished during one episode Γ. Baselines include four
state-of-the-art MARL-based dec-offloading approaches in latest
MEC works, including MDOff (MADDPG-based offloading
[15]), CMOff (COMA-based offloading [18]), MPOff (MAPPO-
based offloading [19]) and MTOff (MATD3-based offloading
[20]). These four baselines are all based on CTDE paradigm
suitable for dec-offloading in MEC where D2D communication
is unavailable. The main differences among them lie in the ways
how offloading policies are learned, e.g., MDOff tends to learn
more deterministic offloading policies for MDs while MPOff
more stochastic, MDOff, MPOff and MTOff support learning
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 7

0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
2
4
6
A v g C o s t
E p i s o d e
M
D O f f  
M
T O f f  
M
P O f f
C M
O f f  
E x p l a b O f f
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
0 . 0
0 . 2
0 . 4
0 . 6
0 . 8
1 . 0
S u c c R a t e
E p i s o d e
(a) MEC with 2 ESs and 3 MDs.
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
2
4
6
A v g C o s t
E p i s o d e
M
D O f f
M
T O f f
M
P O f f
C M
O f f
E x p l a b O f f
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
0 . 0
0 . 2
0 . 4
0 . 6
0 . 8
1 . 0
S u c c R a t e
E p i s o d e
(b) MEC with 2 ESs and 5 MDs.
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
2
4
6
A v g C o s t
E p i s o d e
M
D O f f  
M
T O f f
M
P O f f
C M
O f f
E x p l a b O f f  
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
0 . 0
0 . 2
0 . 4
0 . 6
0 . 8
1 . 0
S u c c R a t e
E p i s o d e
(c) MEC with 3 ESs and 7 MDs.
Fig. 6: Performance comparison between different MARL-based ofﬂoading methods.
both cooperative and competitive offloading policies for MDs
while CMOff only cooperative.
B. Performance Comparison between Dec-Ofﬂoading Methods
In 2ES-3MD (Fig. 6a), ExplabOff is observed to achieve the
lowest AvgCost (around 0.95), as well as the highest converging
speed (converge at about the episode 0.7 × 105). MDOff and
CMOff are seen to achieve close AvgCost as ExplabOff, at
about 1.25), but the converging speed of MDOff is much slower
(at about the episode 2.5 × 105). Meanwhile, the SuccRate of
ExplabOff is also the highest, reaching almost 100% after the
episode 0.7 × 105 (i.e., the convergence point).
When it comes to 2ES-5MD (Fig. 6b), the ES computing
resources competition between ofﬂoaded tasks from MDs is
more intensive. Both MDOff and MTOff are found to be
difﬁcult to reach satisfactory AvgCost, until at about the
episode 7 × 105, but ExplabOff could still converge at about
the episode 1.25×105 with the lowest AvgCost 1.35. Similarly,
the SuccRate of ExplabOff is also much higher than other four
baselines, reaching around 90% after the episode 3.75 × 105.
The performance degradation here compared to the MEC in
Fig. 6a are primarily due to the increase of MDs, competing
more intensively for two BSs’ computing resources.
In 3ES-7MD (Fig. 6c), MDOff and MTOff are noticed
to be hard to achieve low AvgCost and high SuccRate,
possibly resulting from the higher requirement of ofﬂoading
collaboration between MDs as the growth of both #BSs and
#MDs also calls for more careful selection of BS by each MD.
However, ExplabOff keeps achieving the lowest AvgCost and
highest SuccRate, at about 1.2 and 93%, respectively, in com-
parison to four baselines, which shows that the incorporation
of strengthening superior and weakening inferior ofﬂoading
experiences through MI-enhanced reward during learning joint
ofﬂoading behaviors could encourage ExplabOff to explore
more desirable ofﬂoading collaborations of MDs.
C. Evaluation of the Impact of MI-Enhanced Offloading Reward
1) Only strengthen superior MI in B+: We first train ExplabOff
without using INCE and IL1Out as part of the reward (in Eq. (17)),
namely ExplabOff-w/o-MI, then begin to merely strengthen INCE
from the episode 4.5×105, denoted as ExplabOff-INCE, as shown
by the blue and red lines in Fig. 7a, respectively. It is noticed
that, although ExplabOff-w/o-MI previously has converged at
about the episode 0.65 × 105, ExplabOff-INCE still sees obvious
reductions of AvgCost after using INCE. It suggests that superior
offloading collaborations could still be found and exploited by
ExplabOff-INCE to further improve offloading performance even
after the convergence of ExplabOff-w/o-MI.
2) Only weaken inferior MI in B−: Like the above, we
begin to merely weaken IL1Out from the episode 4.5 × 105,
denoted as ExplabOff-IL1Out, as shown by the red line in Fig. 7b.
Similar decline of AvgCost is also observed by ExplabOff-IL1Out,
indicating the effectiveness of breaking out inferior offloading
collaborations in B−to reach higher offloading performances.
3) The impact comparison of whether using MI in ExplabOff:
We further conduct experiments to investigate the impact of
using Neither/INCE/IL1Out/Both on AvgCost, from the beginning
of learning MDs’ ofﬂoading policies, as shown in Fig. 7c. It
can be seen that the ExplabOff (w/o B+&B−) using neither
INCE nor IL1Out (denoted by the green line) could easily fall
into unpromising ofﬂoading collaborations with the highest
AvgCost around 1.95. Whereas, by adopting INCE, IL1Out or
both (denoted by the yellow, blue, red lines, respectively),
more superior and less inferior collaborations can be achieved,
resulting in lower AvgCost compared to using neither INCE
nor IL1Out. Also, the ExplabOff using both INCE and IL1Out
achieves the lowest AvgCost around 1.35. These suggest that
utilizing MI as enhanced reward for MARL-based ofﬂoading
could encourage MDs to explore more collaborated ofﬂoading.
D. Evaluation of the Impact of Distinguished Buffers for MI
In this part, we further empirically evaluate the beneﬁts of
setting distinguished ofﬂoading experience buffers (B+ and
B−) to access the MI of superior and inferior collaborations.
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 8

0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
1
2
3
4
5
A v g C o s t
E p i s o d e
 E x p l a b O f f - w / o - M
I
 E x p l a b O f f - I N C E
s t r e n g t h e n  I N C E  i n  B
+
(a) Strengthen MI in B+ at Episode 4.5×105.
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
1
2
3
4
5
A v g C o s t
E p i s o d e
 E x p l a b O f f - w / o - M
I
 E x p l a b O f f - I L 1 O u t
w e a k e n  I L 1 O u t  i n  B
-
(b) Weaken MI in B−at Episode 4.5×105.
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
1
2
3
4
A v g C o s t
E p i s o d e
 E x p l a b O f f   
 E x p l a b O f f  ( w / o  B
+ )
 E x p l a b O f f  ( w / o  B
- )  
 E x p l a b O f f  ( w / o  B
+ & B
- )
(c) Impact of superior/inferior MI on AvgCost.
Fig. 7: Impact of MI-enhanced ofﬂoading rewards on system average costs.
1) INCE by B+/undistinguished buffer: We save offloading
experiences in B+ every 1 × 104 episodes during training
offloading policies, and get 10 repositories for the initial 1 × 104
episodes indexed from 1 to 10 (as shown by the horizontal axis
in Fig. 8a). We also save the parameters of the INCE estimator,
i.e., ψN, every 1 × 104 episodes and take the saved ψN to
estimate INCE for the 10 repositories (as shown by the vertical
axis in Fig. 8a). It can be seen obviously that the higher the
repository index, the darker is the INCE values (normalized
over all values). By seeing each line from left to right in
Fig. 8a, we can find that new repositories consist of more
superior offloading behaviors than old ones, thus encouraging
and guiding ExplabOff to achieve more collaborated offloading.
In contrast, repositories saved from undistinguished offloading
experiences are hard to see increasing INCE values, indicated by
the nearly unchanged colors (except for the first three buffer
indexes) in each line of Fig. 8b. It also coincides with the
previous observation of the green line in Fig. 7c that ExplabOff
w/o B+&B−converges at about the episode 0.4 × 105 so that
the converged offloading repositories could no longer bring in
darker INCE values afterwards.
Positive buffer index
InfoNCE index
(a) INCE by superior B+.
Buffer index
InfoNCE index
(b) INCE by undistinguished B.
Fig. 8: Impact of distinguished superior B+ on INCE.
2) IL1Out estimation by B−/undistinguished buffer: Similarly,
we save ofﬂoading experiences in B−every 1 × 104 episodes
during training ExplabOff. As seen in Fig. 9a, with the growth
of the B−index, the colors of IL1Out values become weaker and
weaker, indicating that inferior ofﬂoading collaborations are
broken and dis-encouraged with the going of training so that
ExplabOff could escape from falling into undesired ofﬂoading
behaviors. In contrast, the colors of IL1Out in Fig. 9b are much
darker than those in Fig. 9a, showing that inferior collaborations
still exist in the 10-th repository from the undistinguished
buffer and ofﬂoading behaviors could fall into the undesired
collaborations corresponding to the dark color.
Negative buffer index
L1Out index
(a) IL1Out by inferior B−.
Buffer index
L1Out index
(b) IL1Out by undistinguished B.
Fig. 9: Impact of distinguished inferior B−on IL1Out.
E. Evaluation of the Impact of MI Weights on AvgCost
Moreover, we evaluate the impact of the MI weights on
offloading performance through different combinations of µ and
ν, as shown in Fig. 10. Considering the larger color variations
of INCE than IL1Out in the above experiment, we list more weight
values µ for INCE in the combinations. It can be seen that too large
or small values of µ, i.e., µ = 20 and µ = 0.1, achieve the highest
AvgCost, suggesting that too large or small superior-MI reward
µINCE could take too much or little attention from the vanilla
reward in the learning objective of Eq. (17), both of which leading
to undesired offloading performances. Similarly , we find that the
inferior-MI reward νIL1Out should also not overtake focus from
the vanilla reward in Eq. (17). Therefore, we set µ = 3.5, ν = 1
in all the experiments, unless specified otherwise.
0 . 0
2 . 5 x 1 0 5
5 . 0 x 1 0 5
7 . 5 x 1 0 5
1 . 0 x 1 0 6
1 . 5
3 . 0
4 . 5
A v g C o s t
E p i s o d e
u  =  0 . 1 ,  v  =  1  
u  =  1 ,   v  =  1 0
u  =  2 ,     v  =  1  
u  =  3 . 5 ,   v  =  1
u  =  7 ,     v  =  1  
u  =  1 0 ,   v  =  5
u  =  2 0 ,   v  =  1
Fig. 10: Impacts of INCE and IL1Out weights on system average costs.
F. Visualization of Exploration and Collaboration for ExplabOff
In view of the motivation of this work, we further take insights
into the offloading exploration and collaboration ability of Ex-
plabOff by visualizing MDs’ offloading destination distributions
during the initial training period and MDs’ offloading destinations
after convergence, as shown in Fig. 11. In Fig. 11a, we sample
MDs’ offloading destinations (ES 1/2) from 100 episodes of the
initial training period and plot the selection of ES1 and ES2 in
the horizontal and vertical axes, respectively. The darker of one
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 9

point for one color, the more times of one ES selected by one
MD for task offloading. As can be seen, the MDs’ selections of
ESs almost spread all over the left-bottom half of the axes with
nearly similar point-color depth, showing that ExplabOff could
explore offloading collaborations as much as possible during the
initial training period with the help of the MI-enhanced reward.
In Fig. 11b, we separately visualize MDs’ selections of ESs
after training convergence in three MEC environments, i.e., 2ES-
3MD, 2ES-5MD and 3ES-7MD. As can be seen in 2ES-3MD,
MD1 (black line) and MD3 (blue line) are observed to offload
tasks to ES2 and MD2 (red line) offloads to ES1 so that MDs
optimally collaborate their offloading destinations to maximally
utilize edge computing resources. Similarly, In 2ES-5MD, three
MDs with heavy overall workloads are seen to choose ES2 while
two MDs with light overall workloads choose ES1, achieving
maximal utilization of ESs’ resources. Also, similar collaboration
could be found in 3ES-7MD, indicating the efficacy of adopting
MI-enhanced reward to facilitate collaborative offloading.
MD1
MD2
MD3
0    1    2   3   4    5    6    7    8   9   10
Slot  (offloading to ES1)
10
9
8
7
6
5
4
3
2
1
0
Slot  (offloading to ES2)
(a) Exploration visualization.
M
D
1
2
3
4
5
6
7
8
9
1 0
0
1
2
3
1
357
E S
S l o t
0
1
2
1
3
5
E S
0
1
2
1
23
E S
M
E C :  3 E S - 7 M
D
M
E C :  2 E S - 3 M
D
M
E C :  2 E S - 5 M
D
M
D
M
D
(b) Collaboration visualization.
Fig. 11: Visualization of exploration and collaboration for ExplabOff.
G. Real Testbed Experiments
We further conduct experiments on a real testbed, which
is built with Wi-Fi routers (802.11n wireless LAN) as BSs,
workstations as ESs (ES1 and ES2 are equipped with 1× Intel
Core i9-9900K, ES3 is equipped with Intel Xeon Gold 5218),
and Raspberry-Pi-4B (equipped with Cortex-A72 1.5GHz and
also integrated with 802.11n) as MDs, as shown in Fig. 12. As a
popular mobile application, face recognition with four different
picture sizes (22/31/32/64 kb) is taken as MD’s tasks. We build
testbeds with different numbers of BSs and MDs, including 2BS-
3MD, 2BS-5MD, 3BS-5MD, 3BS-7MD. To calculate system
cost for each task, we get the time cost by recording CPU
occupation time and data transmission time and estimate the
energy cost based on the method in [44]. We train ExplabOff,
as well as MDOff, MTOff, MPOff and CMOff, for 20000
episodes on each type of testbed, and report both the AvgCost
and SuccRate of each algorithm after convergence.
In the 2BS-3MD testbed as shown in Fig. 13a, ExplabOff
is noticed to achieve the lowest AvgCost around 2.45, while
CMOff, MDOff, MPOff and MTOff are both higher than 3.2.
Correspondingly, the SuccRate of ExplabOff is seen to reach
close to 1.0, while others are both lower than 0.45. Similar
performance comparisons between ExplabOff and other baselines
are also observed in the 2BS-5MD, 3BS-5MD, 3BS-7MD
testbeds, showing the superiority of ExplabOff over others which
could be resulted from its more explorative and collaborative
offloading ability. In addition, with the growth of #MDs per
Fig. 12: A real testbed to evaluate MARL-based ofﬂoading.
BS in the four types of testbed, i.e., 2BS-3MD in Fig. 13a, 3BS-
5MD in Fig. 13c, 3BS-7MD in Fig. 13d, 2BS-5MD in Fig. 13b,
each individual approach is found to meet obvious increase of
AvgCost, along with the decrease of SuccRate. Possible reasons
mainly lie in the drop of whole available ES computing resources
for each MD from 2BS-3MD to 2BS-5MD, leading to larger
task latency and MD’s energy consumption that inevitably means
performance degradation of both AvgCost and SuccRate.
E x p l a b O f f  C M
O f f
M
D O f f
M
P O f f
M
T O f f
2 . 0
2 . 5
3 . 0
3 . 5
4 . 0
A v g C o s t
 A v g C o s t
0 . 2
0 . 4
0 . 6
0 . 8
1 . 0
S u c c R a t e
 S u c c R a t e
(a) Testbed with 2 BSs and 3 MDs.
E x p l a b O f f  C M
O f f
M
D O f f
M
P O f f
M
T O f f
2
3
4
5
6
A v g C o s t
 A v g C o s t
0 . 0
0 . 2
0 . 4
0 . 6
0 . 8
S u c c R a t e
 S u c c R a t e
(b) Testbed with 2 BSs and 5 MDs.
E x p l a b O f f  C M
O f f
M
D O f f
M
P O f f
M
T O f f
2 . 0
2 . 5
3 . 0
3 . 5
4 . 0
4 . 5
A v g C o s t
 A v g C o s t
0 . 2
0 . 4
0 . 6
0 . 8
1 . 0
 S u c c R a t e
S u c c R a t e
(c) Testbed with 3 BSs and 5 MDs.
E x p l a b O f f  C M
O f f
M
D O f f
M
P O f f
M
T O f f
2
3
4
5
6
7
A v g C o s t
 A v g C o s t
0 . 0
0 . 2
0 . 4
0 . 6
0 . 8
1 . 0
S u c c R a t e
 S u c c R a t e
(d) Testbed with 3 BSs and 7 MDs.
Fig. 13: Performance evaluation on the real testbed.
VI. CONCLUSION
This paper focuses on developing more explorative and col-
laborative MARL-based ofﬂoading approaches for MEC where
D2D communication is unavailable. By consciously exploiting
the implicit exploration and collaboration information involved
in MDs’ ofﬂoading states and actions, we design two metrics
to assess exploration and collaboration for MDs’ ofﬂoading
actions, and assemble them into the MI between MDs’ states
and actions that is further adopted as an additive-reward except
for the vanilla-reward during centralized-training. Moreover,
we distinguish MI between superior and inferior ofﬂoading, and
strengthen and weaken them discriminatively to further improve
ofﬂoading performance. We conduct extensive experiments to
demonstrate the superiority of our approach against state-of-
the-art MARL-based ofﬂoading. Future work could include
the improvement of ExplabOff to promote exploration and
collaboration of MDs in MEC where D2D communication is
available between neighbors, as well as the investigation of
ExplabOff in more large-scale MEC environments.
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---

## Page 10

REFERENCES
[1] Q. Tang, F. R. Yu, R. Xie, and et al., “Internet of intelligence: A survey on
the enabling technologies, applications, and challenges,” IEEE Commun.
Surveys Tuts., vol. 24, no. 3, pp. 1394–1434, 2022.
[2] P. Porambage, J. Okwuibe, M. Liyanage, M. Ylianttila, and T. Taleb,
“Survey on multi-access edge computing for internet of things realization,”
IEEE Commun. Surveys Tuts., vol. 20, no. 4, pp. 2961–2991, 2018.
[3] P. Mach and Z. Becvar, “Mobile edge computing: A survey on architecture
and computation ofﬂoading,” IEEE Commun. Surveys Tuts., vol. 19, no. 3,
pp. 1628–1656, 2017.
[4] X. Wang, J. Ye, and J. C. Lui, “Decentralized task ofﬂoading in
edge computing: A multi-user multi-armed bandit approach,” in IEEE
Conference on Computer Communications.
IEEE, 2022, pp. 1199–1208.
[5] N. Eshraghi and B. Liang, “Joint ofﬂoading decision and resource allo-
cation with uncertain task computing requirement,” in IEEE Conference
on Computer Communications.
IEEE, 2019, pp. 1414–1422.
[6] J. Ren, J. Liu, Y. Zhang, and et al., “An efﬁcient two-layer task ofﬂoading
scheme for mec system with multiple services providers,” in IEEE
Conference on Computer Communications.
IEEE, 2022, pp. 1519–
1528.
[7] H. Djigal, J. Xu, L. Liu, and Y. Zhang, “Machine and deep learning for
resource allocation in multi-access edge computing: A survey,” IEEE
Commun. Surveys Tuts., 2022.
[8] A. Shakarami, M. Ghobaei-Arani, and A. Shahidinejad, “A survey on
the computation ofﬂoading approaches in mobile edge computing: A
machine learning-based perspective,” Computer Networks, vol. 182, p.
107496, 2020.
[9] L. Lin, X. Liao, H. Jin, and P. Li, “Computation ofﬂoading toward edge
computing,” Proceedings of the IEEE, vol. 107, no. 8, pp. 1584–1607,
2019.
[10] Z. Zabihi, A. M. Eftekhari Moghadam, and M. H. Rezvani, “Reinforce-
ment learning methods for computation ofﬂoading: A systematic review,”
ACM Computing Surveys, vol. 56, no. 1, pp. 1–41, 2023.
[11] L. Huang, S. Bi, and Y. J. Zhang, “Deep reinforcement learning for online
computation ofﬂoading in wireless powered mobile-edge computing
networks,” IEEE Trans. Mobile Comput., 2019.
[12] P. Dai, K. Hu, X. Wu, H. Xing, and Z. Yu, “Asynchronous deep rein-
forcement learning for data-driven task ofﬂoading in MEC-empowered
vehicular networks,” in IEEE Conference on Computer Communications.
IEEE, 2021, pp. 1–10.
[13] T. Li, K. Zhu, N. C. Luong, and et al., “Applications of multi-agent
reinforcement learning in future internet: A comprehensive survey,” IEEE
Commun. Surveys Tuts., vol. 24, no. 2, pp. 1240–1279, 2022.
[14] J. Tan, R. Khalili, H. Karl, and A. Hecker, “Multi-agent distributed
reinforcement learning for making decentralized ofﬂoading decisions,”
in IEEE Conference on Computer Communications.
IEEE, 2022, pp.
2098–2107.
[15] H. Peng and X. Shen, “Multi-agent reinforcement learning based resource
management in MEC-and UAV-assisted vehicular networks,” IEEE J.
Sel. Areas Commun., vol. 39, no. 1, pp. 131–141, 2021.
[16] S. Guo and X. Zhao, “Multi-agent deep reinforcement learning based
transmission latency minimization for delay-sensitive cognitive satellite-
uav networks,” IEEE Trans. Commun., vol. 71, no. 1, pp. 131–144,
2022.
[17] Z. Gao, L. Yang, and Y. Dai, “Large-scale computation ofﬂoading using
a multi-agent reinforcement learning in heterogeneous multi-access edge
computing,” IEEE Trans. Mobile Comput., vol. 22, no. 6, pp. 3425–3443,
2023.
[18] C. Liu, F. Tang, Y. Hu, and et al., “Distributed task migration optimization
in MEC by extending multi-agent deep reinforcement learning approach,”
IEEE Trans. Parallel Distrib. Syst., vol. 32, no. 7, pp. 1603–1614, 2021.
[19] W. Liu, B. Li, W. Xie, Y. Dai, and Z. Fei, “Energy efﬁcient computation
ofﬂoading in aerial edge networks with multi-agent cooperation,” IEEE
Trans. Wireless Commun., vol. 22, no. 9, pp. 5725–5739, 2023.
[20] N. Zhao, Z. Ye, Y. Pei, and et al., “Multi-agent deep reinforcement
learning for task ofﬂoading in UAV-assisted mobile edge computing,”
IEEE Trans. Wireless Commun., vol. 21, no. 9, pp. 6949–6960, 2022.
[21] A. Sacco, F. Esposito, G. Marchetto, and P. Montuschi, “Sustainable
task ofﬂoading in UAV networks via multi-agent reinforcement learning,”
IEEE Trans. Veh. Technol., vol. 70, no. 5, pp. 5003–5015, 2021.
[22] H. Gao, X. Wang, W. Wei, A. Al-Dulaimi, and Y. Xu, “Com-DDPG: task
ofﬂoading based on multiagent reinforcement learning for information-
communication-enhanced mobile edge computing in the internet of
vehicles,” IEEE Trans. Veh. Technol., vol. 73, no. 1, pp. 348–361, 2024.
[23] Z. Qin, H. Yao, T. Mai, and et al., “Multi-agent reinforcement learning
aided computation ofﬂoading in aerial computing for the internet-of-
things,” IEEE Trans. Serv. Comput., vol. 16, no. 3, pp. 1976–1986,
2023.
[24] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with
contrastive predictive coding,” arXiv preprint arXiv:1807.03748, pp. 1–
13, 2018.
[25] B. Poole, S. Ozair, A. Van Den Oord, A. Alemi, and G. Tucker, “On
variational bounds of mutual information,” in International Conference
on Machine Learning.
PMLR, 2019, pp. 5171–5180.
[26] Y. Zhan, S. Guo, P. Li, and J. Zhang, “A deep reinforcement learning
based ofﬂoading game in edge computing,” IEEE Trans. Comput., vol. 69,
no. 6, pp. 883–893, 2020.
[27] F. Meng, P. Chen, L. Wu, and J. Cheng, “Power allocation in multi-user
cellular networks: Deep reinforcement learning approaches,” IEEE Trans.
Wireless Commun., vol. 19, no. 10, pp. 6255–6267, 2020.
[28] P. Dent, G. E. Bottomley, and T. Croft, “Jakes fading model revisited,”
Electronics Letters, vol. 13, no. 29, pp. 1162–1163, 1993.
[29] Y. S. Nasir and D. Guo, “Multi-agent deep reinforcement learning for
dynamic power allocation in wireless networks,” IEEE J. Sel. Areas
Commun., vol. 37, no. 10, pp. 2239–2250, 2019.
[30] A. Younis, T. X. Tran, and D. Pompili, “Energy-efﬁcient resource
allocation in c-rans with capacity-limited fronthaul,” IEEE Trans. Mobile
Comput., vol. 20, no. 2, pp. 473–487, 2021.
[31] J. Lou, H. Luo, Z. Tang, W. Jia, and W. Zhao, “Efﬁcient container
assignment and layer sequencing in edge computing,” IEEE Trans. on
Serv. Comput., vol. 16, no. 2, pp. 1118–1131, 2022.
[32] J. Lou, Z. Tang, W. Jia, W. Zhao, and J. Li, “Startup-aware dependent
task scheduling with bandwidth constraints in edge computing,” IEEE
Trans. Mobile Comput., vol. 23, no. 2, pp. 1586–1600, 2024.
[33] T. Zhu, J. Li, Z. Cai, Y. Li, and H. Gao, “Computation scheduling for
wireless powered mobile edge computing networks,” in IEEE Conference
on Computer Communications.
IEEE, 2020, pp. 596–605.
[34] J. N. Hooker, “Chapter 15 - operations research methods in constraint
programming,” in Handbook of Constraint Programming, ser. Foundations
of Artiﬁcial Intelligence.
Elsevier, 2006, vol. 2, pp. 527–570.
[35] O. Hern´andez-Lerma and J. B. Lasserre, Discrete-time Markov control
processes: Basic optimality criteria.
Springer Science & Business
Media, 2012, vol. 30.
[36] T. Haarnoja, H. Tang, P. Abbeel, and S. Levine, “Reinforcement learning
with deep energy-based policies,” in International Conference on Machine
Learning.
PMLR, 2017, pp. 1352–1361.
[37] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-
policy maximum entropy deep reinforcement learning with a stochastic
actor,” in International Conference on Machine Learning.
PMLR, 2018,
pp. 1861–1870.
[38] D. Ye and Z. Lu, “Mutual-information regularized multi-agent policy
iteration,” Advances in Neural Information Processing Systems, vol. 36,
2024.
[39] W. Kim, W. Jung, M. Cho, and Y. Sung, “A variational approach to mutual
information-based coordination for multi-agent reinforcement learning,”
in International Conference on Autonomous Agents and Multiagent
Systems, 2023, pp. 40–48.
[40] P. Li, H. Tang, T. Yang, and et al., “Pmic: Improving multi-agent
reinforcement learning with progressive mutual information collaboration,”
in International Conference on Machine Learning.
PMLR, 2022, pp.
12 979–12 997.
[41] L. Paninski, “Estimation of entropy and mutual information,” Neural
Computation, vol. 15, no. 6, pp. 1191–1253, 2003.
[42] F. Bai and A. Helmy, “A survey of mobility models,” Wireless Adhoc
Networks, vol. 206, pp. 147–176, 2004.
[43] M. Ulm, P. Widhalm, and N. Br¨andle, “Characterization of mobile phone
localization errors with OpenCellID data,” in International Conference
on Advanced Logistics and Transport.
IEEE, 2015, pp. 100–104.
[44] Y. Wen, W. Zhang, and H. Luo, “Energy-optimal mobile application
execution: Taming resource-poor mobile devices with cloud clones,” in
IEEE Conference on Computer Communications.
IEEE, 2012, pp.
2716–2720.
Authorized licensed use limited to: Shanghai Jiaotong University. Downloaded on May 07,2026 at 13:03:01 UTC from IEEE Xplore.  Restrictions apply.

---
